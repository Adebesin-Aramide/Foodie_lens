import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time

def get_images_from_google(wd, delay, max_images):
    def scroll_down(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)
    
    image_urls = set()
    skips = 0
    
    while len(image_urls) + skips < max_images:
        scroll_down(wd)
        
        thumbnails = wd.find_elements(By.CSS_SELECTOR, "div.H8Rx8c")
    
        
        for img in thumbnails[len(image_urls) + skips:max_images]:
            try:
                img.click()
                time.sleep(delay)
            except:
                continue
            
            images = wd.find_elements(By.CSS_SELECTOR, ".sFlh5c, .pT0Scc, .iPVvYb")
            for image in images:
                if image.get_attribute("src") in image_urls:
                    max_images += 1
                    skips += 1
                    break
                
                if image.get_attribute("src") and "http" in image.get_attribute("src"):
                    image_urls.add(image.get_attribute("src"))
                    print(f"Found {len(image_urls)}")
                    
    return image_urls


def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        
        if image.format not in ["JPEG", "PNG"]:
            print(f"Skipping image with unsupported format: {url}")
            return
        
        file_path = os.path.join(download_path, file_name)
    
        with open(file_path, "wb") as f:
            image.save(f, "JPEG")
        
        print("Successfully downloaded image")
    except Exception as e:
        print("Failed to download image", e)
        

search_query = input("Enter search query: ")

download_path = "C:\\Users\\aramide\\Documents\\Foodie_lens\\food_dataset\\egusi\\"
os.makedirs(download_path, exist_ok=True)

options = Options()
options.add_argument("--start-maximized")
wd = webdriver.Chrome(options=options)

search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
wd.get(search_url)

urls = get_images_from_google(wd, 1, 5)

for i, url in enumerate(urls):
    download_image(download_path, url, str(i) + ".jpg")
    
wd.quit()