import msgpack
import numpy as np
import pickle
import logging

logger = logging.getLogger("SerializationUtils")

def _msgpack_numpy_encoder(obj):
    """Custom encoder for Numpy types into Msgpack-compatible format."""
    if isinstance(obj, np.ndarray):
        return {
            b'__nd__': True,
            b'data': obj.tobytes(),
            b'dtype': str(obj.dtype),
            b'shape': obj.shape
        }
    elif isinstance(obj, (np.bool_, np.number)):
        return obj.item()
    return obj

def _msgpack_numpy_decoder(obj):
    """Custom decoder for Numpy types from Msgpack-compatible format."""
    if b'__nd__' in obj:
        return np.frombuffer(obj[b'data'], dtype=obj[b'dtype']).reshape(obj[b'shape'])
    return obj

def pack(data, use_msgpack=True):
    """
    Serializes data using Msgpack (default) or Pickle.
    """
    if use_msgpack:
        try:
            # We prefix with a marker byte to distinguish from Pickle if needed, 
            # but usually we handle it at the decoder level.
            return msgpack.packb(data, default=_msgpack_numpy_encoder, use_bin_type=True)
        except Exception as e:
            logger.error(f"❌ Msgpack pack error: {e}. Falling back to Pickle.")
            return pickle.dumps(data)
    else:
        return pickle.dumps(data)

def unpack(data):
    """
    Deserializes data, automatically detecting if it's Msgpack or Pickle.
    Pickle usually starts with b'\x80' (version marker) or b'(' .
    Msgpack fixmap/fixarray starts with specific ranges.
    """
    if not data:
        return None
        
    # Heuristic: Pickle version 2+ starts with \x80
    if data.startswith(b'\x80'):
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"❌ Pickle unpack failed: {e}")
            
    # Try Msgpack
    try:
        return msgpack.unpackb(data, object_hook=_msgpack_numpy_decoder, raw=False)
    except Exception as e:
        # If it wasn't msgpack, try pickle as last resort
        try:
            return pickle.loads(data)
        except:
             logger.error(f"❌ Comprehensive unpack failure.")
             return None
