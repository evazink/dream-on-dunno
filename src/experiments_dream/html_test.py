import time
from playwright.sync_api import sync_playwright

def html_to_png(html_str, output_path="output.png", width=800, height=600):
    start_time = time.time()
    print(f"[{time.time() - start_time:.4f}s] Starting html_to_png function...")

    with sync_playwright() as p:
        print(f"[{time.time() - start_time:.4f}s] Playwright context entered. Launching browser...")
        browser = p.chromium.launch()
        print(f"[{time.time() - start_time:.4f}s] Browser launched. Creating new page...")
        page = browser.new_page(viewport={"width": width, "height": height})
        print(f"[{time.time() - start_time:.4f}s] Page created. Setting content...")
        page.set_content(html_str)
        print(f"[{time.time() - start_time:.4f}s] Content set. Taking screenshot...")
        page.screenshot(path=output_path, full_page=True)
        print(f"[{time.time() - start_time:.4f}s] Screenshot taken. Closing browser...")
        browser.close()
        print(f"[{time.time() - start_time:.4f}s] Browser closed. Exiting Playwright context.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[{total_time:.4f}s] html_to_png function finished. Total time: {total_time:.4f} seconds.")

# Example usage
html = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: sans-serif; background: #f0f0f0; }
    h1 { color: tomato; text-align: center; padding-top: 200px; }
  </style>
</head>
<body>
  <h1>Hello, image!</h1>
</body>
</html>
"""

print("\n--- Running html_to_png example ---")
html_to_png(html, "example_output.png")
print("--- Example finished ---\n")