import markdown
import asyncio
from playwright.async_api import async_playwright
import os

# Configuration
INPUT_MD = 'resume.md'
INPUT_CSS = 'resume.css'
OUTPUT_HTML = 'resume_preview.html'
OUTPUT_PDF = 'resume.pdf'

def md_to_html(md_path, css_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert Markdown to HTML with tables and extra features
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])
    
    with open(css_path, 'r', encoding='utf-8') as f:
        css_content = f.read()
    
    # Wrap in full HTML structure with CSS
    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Resume - 方帅</title>
        <style>
            {css_content}
        </style>
    </head>
    <body>
        <div class="resume-container">
            {html_content}
        </div>
    </body>
    </html>
    """
    return full_html

async def html_to_pdf(html_content, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Set content directly
        await page.set_content(html_content)
        
        # Wait for fonts to load if any
        await page.wait_for_load_state('networkidle')
        
        # Print to PDF
        # Scale can be adjusted if needed, 0.8 or 0.9 is often good for fit
        await page.pdf(
            path=pdf_path,
            format='A4',
            print_background=True,
            margin={
                'top': '20px',
                'bottom': '20px',
                'left': '20px',
                'right': '20px'
            },
            display_header_footer=False,
            prefer_css_page_size=True
        )
        
        await browser.close()

async def main():
    print(f"Reading {INPUT_MD}...")
    if not os.path.exists(INPUT_MD):
        print(f"Error: {INPUT_MD} not found!")
        return

    print("Converting Markdown to HTML...")
    html_content = md_to_html(INPUT_MD, INPUT_CSS)
    
    # Save a preview HTML for debugging
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Preview saved to {OUTPUT_HTML}")

    print("Generating PDF using Playwright...")
    await html_to_pdf(html_content, OUTPUT_PDF)
    print(f"Success! PDF saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    asyncio.run(main())
