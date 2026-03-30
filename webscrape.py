# work with this file !!

import os
import json
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://incidecoder.com"
OUTPUT_PATH = "scraped_products/all_products.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def load_existing(output_path):
    """Load already-scraped products and return (list, url_set) for resuming."""
    if not os.path.exists(output_path):
        return [], set()
    with open(output_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    return products, {p["product_url"] for p in products if "product_url" in p}


def save(products, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)


def scrape_product(session, url):
    resp = session.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    product = {
        "product_url": url,
        "product_name": "",
        "brand": "",
        "ingredients": "",
        "product_details": "",
        "image_url": "",
    }

    # Product name
    name = soup.select_one("span#product-title")
    if name:
        product["product_name"] = name.get_text(strip=True)

    # Brand
    brand = soup.select_one("span#product-brand-title a")
    if brand:
        product["brand"] = brand.get_text(strip=True)

    # Ingredients
    ingred_div = soup.select_one("div#ingredlist-short")
    if ingred_div:
        ingreds = [a.get_text(strip=True) for a in ingred_div.select("a.ingred-link")]
        product["ingredients"] = ", ".join(ingreds)

    # Product details — freeform text with skin type / concerns info
    details = soup.select_one("span#product-details")
    if details:
        product["product_details"] = details.get_text(separator=" ", strip=True)

    # Image
    for selector in ["img#product-image", "img.product-image", "img[src*='product']"]:
        img = soup.select_one(selector)
        if img:
            product["image_url"] = img.get("src", "")
            break

    return product


def main():
    all_products, scraped_urls = load_existing(OUTPUT_PATH)
    if all_products:
        print(f"Resuming: {len(all_products)} products already scraped.\n")

    session = requests.Session()
    total_scraped = len(all_products)
    next_url = f"{BASE_URL}/products/all"
    page_num = 0

    while next_url:
        page_num += 1
        print(f"[Page {page_num}] Fetching: {next_url}")
        try:
            resp = session.get(next_url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"  Error fetching page: {e}")
            time.sleep(3)
            continue

        # Collect product URLs from this page
        page_urls = []
        for link in soup.select("a.simpletextlistitem"):
            href = link.get("href", "")
            if href.startswith("/products/"):
                product_url = f"{BASE_URL}{href}"
                if product_url not in scraped_urls:
                    page_urls.append(product_url)

        print(f"  {len(page_urls)} new products on this page")

        # Scrape each product on this page immediately
        for url in page_urls:
            try:
                product = scrape_product(session, url)
                if product:
                    all_products.append(product)
                    scraped_urls.add(url)
                    total_scraped += 1
                    print(f"  [{total_scraped}] {product['brand']} — {product['product_name']}")

                # Checkpoint every 100 products
                if total_scraped % 100 == 0:
                    save(all_products, OUTPUT_PATH)
                    print(f"  >>> Checkpoint saved: {total_scraped} total\n")

                time.sleep(0.5)

            except Exception as e:
                print(f"  Error scraping {url}: {e}")

        # Follow "Next page >>" link
        next_link = soup.find("a", string=re.compile(r"Next page", re.I))
        next_url = f"{BASE_URL}{next_link['href']}" if next_link and next_link.get("href") else None

        # Save at end of each page and pause before next listing page
        save(all_products, OUTPUT_PATH)
        time.sleep(1)

    # Final save
    save(all_products, OUTPUT_PATH)
    print(f"\nDone. Total products scraped: {len(all_products)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
