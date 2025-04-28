import os
import json
from playwright.sync_api import sync_playwright


def scrape_text_content(page, url, elements='h1, h2, h3, p'):
    page.goto(url)
    page.wait_for_load_state('networkidle')
    content = page.locator(elements).all_text_contents()
    return content


def scrape_images_from_url(page, url):
    page.goto(url)
    page.wait_for_load_state('networkidle')
    images = page.locator('img')
    return [images.nth(i).get_attribute('src') for i in range(images.count()) if images.nth(i).get_attribute('src')]


def scrape_expandable_services(page, url, button_selector='.frame.cursor.none', title_selector='h3', desc_selector='p'):
    page.goto(url)
    page.wait_for_load_state('networkidle')
    buttons = page.locator(button_selector)
    for i in range(buttons.count()):
        try:
            buttons.nth(i).click()
            page.wait_for_timeout(300)
        except:
            continue
    titles = page.locator(title_selector).all_text_contents()
    descriptions = page.locator(desc_selector).all_text_contents()
    return titles + descriptions


def save_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def scrape_company_info(page, url='https://34ml.com/company/', save_path='data/company_info.json'):
    company_info = scrape_text_content(page, url)
    save_json({"company_info": company_info}, save_path)


def scrape_services_info(page, url='https://34ml.com/services/', save_path='data/services.json'):
    services = scrape_expandable_services(page, url)
    save_json({"services": services}, save_path)


def scrape_customer_base(page, url='https://34ml.com/work/', save_path='data/customer_base.json'):
    customer_text = scrape_text_content(page, url)
    image_urls = scrape_images_from_url(page, url)
    save_json({"customer_base": customer_text, "images": image_urls}, save_path)


def scrape_blog_articles(page, url='https://34ml.com/blog/', save_path='data/blog_articles.json'):
    blog_text = scrape_text_content(page, url)
    save_json({"blog_articles": blog_text}, save_path)


def scrape_tone_of_voice(page, urls, save_path='data/tone_of_voice.json'):
    tone_data = []
    for url in urls:
        tone_data.extend(scrape_text_content(page, url))
    save_json({"tone_of_voice": tone_data}, save_path)


def main():
    tone_urls = [
        'https://34ml.com/company/',
        'https://34ml.com/services/',
        'https://34ml.com/work/',
        'https://34ml.com/blog/'
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Modular calls
        scrape_company_info(page)
        scrape_services_info(page)
        scrape_customer_base(page)
        scrape_blog_articles(page)
        scrape_tone_of_voice(page, tone_urls)

        browser.close()


if __name__ == '__main__':
    main()
    print("Scraping completed.")