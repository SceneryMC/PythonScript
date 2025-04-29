import io
import re
import time
import requests
from multiprocessing.dummy import Pool
from path_cross_platform import * # Assuming this handles path separators correctly
from secret import idohae_base_path

# --- Keep attribute_map and other functions as they are ---
hubble_attributes = {'folder': 'Hubble', 'name': 'hubble',
                     'source': 'https://esahubble.org/media/archives/images/original',
                     'suffix': "https://esahubble.org/images",
                     'basename': "https://esahubble.org/images/page"}
eso_attributes = {'folder': 'ESO', 'name': 'eso',
                  'source': 'https://www.eso.org/public/archives/images/original',
                  'suffix': "https://www.eso.org/public/images",
                  'basename': "https://www.eso.org/public/images/list"}
attribute_map = {"eso": eso_attributes, "hubble": hubble_attributes}


def get_image_urls(url):
    while True:
        try:
            # Consider adding a timeout
            html_text = requests.get(url, timeout=30).text
            break
        except requests.exceptions.RequestException as e: # Catch broader exceptions
            print(f"Error getting URLs from {url}: {e}. Retrying...")
            time.sleep(5)
    # Use try-except for regex search in case pattern not found
    try:
        url_text = re.search(r'var images.*?</script><div class="image-list image-list-\d+', html_text, re.DOTALL).group()
        ls = re.findall(rf"id: '(.*?)',", url_text)
        return ls
    except AttributeError:
        print(f"Could not find image data pattern on {url}")
        return []


def get_total(url):
    while True:
        try:
             # Consider adding a timeout
            html_text = requests.get(url, timeout=30).text
            break
        except requests.exceptions.RequestException as e: # Catch broader exceptions
            print(f"Error getting total from {url}: {e}. Retrying...")
            time.sleep(5)
    # Use try-except for regex search
    try:
        result = re.search(r"Showing 1 to (\d+) of (\d+)", html_text)
        return int(result.group(1)), int(result.group(2))
    except (AttributeError, ValueError):
         print(f"Could not parse total/per_page from {url}")
         return 0, 0 # Return default values

def get_suffix(image):
    url = f"{attributes['suffix']}/{image}"
    while True:
        try:
            # Consider adding a timeout
            html_text = requests.get(url, timeout=30).text
            break
        except requests.exceptions.RequestException as e: # Catch broader exceptions
            print(f"Error getting suffix page {url}: {e}. Retrying...")
            time.sleep(5)
    # Use try-except for string find
    try:
        # Make finding suffix more robust maybe? Check page structure.
        # Example: Search for download links and parse extension
        # This is a simplified version based on your original code
        suffix_index = html_text.find("Fullsize Original")
        if suffix_index != -1:
            # Look backwards for the extension within the link tag typically preceding it
            link_start = html_text.rfind('<a href=', 0, suffix_index)
            if link_start != -1:
                 link_end = html_text.find('>', link_start)
                 if link_end != -1:
                      link_text = html_text[link_start:link_end]
                      match = re.search(r'\.(tif|jpg|png)\b', link_text, re.IGNORECASE)
                      if match:
                           return match.group(1).lower() # Return common extensions like tif, jpg, png
        print(f"Could not reliably determine suffix for {image} from {url}. Trying common ones.")
        # Fallback or raise error if suffix essential
        # Let's try HEAD requests for common types as a fallback (more robust)
        for ext in ['tif', 'jpg']:
            head_url = f"{attributes['source']}/{image}.{ext}"
            try:
                r_head = requests.head(head_url, timeout=10, allow_redirects=True)
                if r_head.status_code == 200:
                    print(f"Found suffix '{ext}' via HEAD request for {image}")
                    return ext
            except requests.exceptions.RequestException:
                continue # Try next extension
        print(f"ERROR: Failed to find suffix for {image} even with HEAD requests.")
        return None # Indicate failure
    except Exception as e:
        print(f"Error processing suffix page {url}: {e}")
        return None # Indicate failure

# --- download_image_thread now returns data or None on error ---
def download_image_thread(image, suffix, segment):
    headers = {"range": f"bytes={segment[0]}-{segment[1]}"}
    url = f"{attributes['source']}/{image}.{suffix}"
    retries = 3
    for attempt in range(retries):
        try:
            # Use a session potentially, add timeout
            r = requests.get(url, headers=headers, stream=True, timeout=60) # Longer timeout for download
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Download data into memory
            content = r.content # Read all content for this chunk
            print(f"Thread downloaded bytes {segment[0]}-{segment[1]} for {image}.{suffix}")
            return segment[0], content # Return start position and data
        except requests.exceptions.RequestException as e:
            print(f"Error downloading segment {segment} for {url}: {e}. Attempt {attempt + 1}/{retries}")
            if attempt == retries - 1:
                print(f"Thread failed to download segment {segment} for {url} after {retries} attempts.")
                return segment[0], None # Indicate failure for this chunk
            time.sleep(5 + attempt * 2) # Exponential backoff
    return segment[0], None # Should not be reached if retries=3, but for safety

# --- download_image now orchestrates and writes file at the end ---
def download_image(image, suffix):
    if suffix is None:
        print(f"Skipping {image} because suffix could not be determined.")
        with open(f"results/skipped_{attributes['name']}.txt", 'a') as f:
             f.write(f"{attributes['source']}/{image}.(unknown suffix)\n")
        return False

    print(f"Checking size for {image}.{suffix}...")
    source_url = f"{attributes['source']}/{image}.{suffix}"
    try:
        # Use HEAD request with timeout
        r = requests.head(source_url, timeout=30, allow_redirects=True) # Allow redirects for HEAD
        r.raise_for_status()
        # Check if content-length exists and is valid
        content_length_str = r.headers.get('content-length')
        if content_length_str is None:
            print(f"Warning: Content-Length header missing for {source_url}. Attempting download anyway.")
            # Decide how to handle - maybe try download without chunking or skip?
            # For now, let's try to proceed but without size check / chunking logic might need adjustment
            # A simple download without chunking might be safer here:
            # return download_image_simple(image, suffix) # Implement this fallback if needed
            image_size = None # Indicate unknown size
        else:
             image_size = int(content_length_str)

        if image_size is not None and image_size > 96 * 1024 ** 2:
            print(f"TOO LARGE: SIZE = {image_size / 1024 ** 2:.2f} MB! Skipping {image}.{suffix}")
            with open(f"results/skipped_{attributes['name']}.txt", 'a') as f:
                f.write(f"{source_url}\n")
            return True  # 跳过并记录的图片也视作下载成功

        if image_size is None or image_size == 0:
            print(f"Warning: Image size reported as 0 or unknown for {source_url}. Skipping segmented download possibility or trying simple download.")
            # Optionally implement a simple, non-chunked download here
            # For now, we'll skip if size is 0, or proceed cautiously if None
            if image_size == 0:
                print(f"Skipping {image}.{suffix} due to reported size 0.")
                with open(f"results/skipped_{attributes['name']}.txt", 'a') as f:
                    f.write(f"{source_url} (Size: 0)\n")
                return False
            # If size is None, we proceed but the segmentation might not work as intended without total size
            # Let's define a default chunk size maybe? Or better, skip chunking if size is None.
            # Sticking to original logic path requires size, so let's report error if None after check.
            if image_size is None:
                print(f"ERROR: Cannot perform segmented download for {image}.{suffix} without Content-Length. Skipping.")
                with open(f"results/skipped_{attributes['name']}.txt", 'a') as f:
                    f.write(f"{source_url} (Size: Unknown)\n")
                return False


    except requests.exceptions.RequestException as e:
        print(f"Failed to get HEAD for {source_url}: {e}. Skipping image.")
        with open(f"results/skipped_{attributes['name']}.txt", 'a') as f:
             f.write(f"{source_url} (HEAD request failed: {e})\n")
        return False

    # Proceed with chunked download
    print(f"Downloading {image}.{suffix} (Size: {image_size / 1024**2:.2f} MB)...")
    start, end, step = 0, image_size, 8 * 1024 ** 2 # 8MB chunks
    image_segments = [(start, min(start + step, end) - 1) for start in range(0, end, step)]

    pool = Pool() # Use default number of threads (usually based on CPU cores)
    results = []
    for segment in image_segments:
        results.append(pool.apply_async(download_image_thread, args=(image, suffix, segment)))

    pool.close()
    pool.join()

    # Collect results and check for failures
    downloaded_chunks = {}
    all_successful = True
    for res in results:
        try:
            start_pos, data = res.get() # Get result from thread
            if data is None:
                all_successful = False
                print(f"A thread failed for image {image}.{suffix}")
                # break # Option: Stop processing this image if one chunk fails
            else:
                downloaded_chunks[start_pos] = data
        except Exception as e: # Catch potential exceptions from thread execution itself
             print(f"Error getting result from thread for {image}.{suffix}: {e}")
             all_successful = False
             # break

    if not all_successful or len(downloaded_chunks) != len(image_segments):
        print(f"Download failed for {image}.{suffix} because some chunks were not retrieved.")
        # Clean up potentially created file? Or leave it? Decide policy.
        # Maybe log to a failed list instead of skipped list
        with open(f"results/failed_{attributes['name']}.txt", 'a') as f:
             f.write(f"{source_url} (Chunk download failed)\n")
        return False # Indicate failure

    # If all chunks downloaded successfully, write the file
    output_path = path_fit_platform(rf'{idohae_base_path}\{attributes["folder"]}\{image}.{suffix}')
    print(f"All chunks received for {image}.{suffix}. Writing to {output_path}")
    try:
        with open(output_path, "wb") as f:
            # Write chunks in order based on their starting position
            for start_pos in sorted(downloaded_chunks.keys()):
                f.write(downloaded_chunks[start_pos])
        print(f"Successfully saved {output_path}")
        return True
    except IOError as e:
        print(f"Error writing final file {output_path}: {e}")
        # Clean up the potentially partially written file
        try:
            os.remove(output_path)
        except OSError:
            pass
        with open(f"results/failed_{attributes['name']}.txt", 'a') as f:
             f.write(f"{source_url} (File write error: {e})\n")
        return False


def get_download_list(downloaded):
    base_list_url = f"{attributes['basename']}/1/?sort=-release_date"
    image_per_page, total = get_total(base_list_url)
    print(f"Images per page: {image_per_page}, Total images: {total}")

    if total == 0:
        print("Error: Could not determine total number of images. Aborting.")
        return []
    if image_per_page == 0:
        print("Error: Could not determine images per page. Assuming 20 as a fallback.") # Or handle differently
        image_per_page = 20 # Example fallback

    will_download_count = total - downloaded
    if will_download_count <= 0:
        print("No new images to download.")
        return []

    print(f"Need to download {will_download_count} new images.")
    images_to_fetch = []
    # Calculate pages needed carefully
    # If will_download_count is 50, image_per_page is 24:
    # Need page 1 (gets latest 24)
    # Need page 2 (gets next 24, total 48)
    # Need page 3 (gets next 24, need first 2 from this page)
    # Total pages to check = ceil(will_download_count / image_per_page)
    import math
    pages_to_fetch = math.ceil(will_download_count / image_per_page)

    print(f"Fetching image lists from {pages_to_fetch} pages...")

    all_fetched_ids = []
    for i in range(1, pages_to_fetch + 1):
        page_url = f"{attributes['basename']}/{i}/?sort=-release_date"
        ids_on_page = get_image_urls(page_url)
        if not ids_on_page:
            print(f"Warning: No image IDs found on page {i} ({page_url}). Stopping list collection.")
            break # Or continue cautiously? If a page fails, we might miss images.
        all_fetched_ids.extend(ids_on_page)
        print(f"Page {i} list collected ({len(ids_on_page)} IDs). Total IDs so far: {len(all_fetched_ids)}")
        time.sleep(0.5) # Be polite to the server

    # The logic needs images sorted newest first, and we want the oldest *of the new ones*.
    # `all_fetched_ids` contains IDs from page 1 (newest), page 2, ... page N (older)
    # We need the `will_download_count` images starting from the end of the already `downloaded` ones.
    # Example: total=100, downloaded=80. will_download=20.
    # We fetch page 1 (ids 100..77), page 2 (ids 76..53) etc.
    # `all_fetched_ids` = [id100, id99, ..., id77, id76, ..., id53, ...]
    # We need the 20 images that come *after* the 80th image (when sorted by release date).
    # In the list sorted newest-first, these are indices 80 through 99.
    # However, our `get_image_urls` likely gets them in the order they appear on the page. Assuming that order is newest first:
    # The `all_fetched_ids` list contains the newest images first.
    # We need the `will_download_count` images from this list.
    # But we want to download them oldest-first? Your original code reverses the list slice.

    # Let's clarify the desired download order. If you want to download the *newest* images first from the ones you are missing:
    # return all_fetched_ids[:will_download_count]

    # If you want to download the *oldest* images first from the ones you are missing (as your original slice `[::-1]` suggested):
    needed_ids = all_fetched_ids[:will_download_count]
    return needed_ids[::-1] # Reverse to process oldest-new first


if __name__ == '__main__':
    site = input("网站 (hubble/eso)：").lower()
    if site not in attribute_map:
        print("无效的网站名称。请输入 'hubble' 或 'eso'.")
        exit()
    attributes = attribute_map[site]

    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    # Ensure target image directory exists
    target_folder = path_fit_platform(rf'{idohae_base_path}\{attributes["folder"]}')
    if not os.path.exists(target_folder):
         try:
            os.makedirs(target_folder)
            print(f"Created target directory: {target_folder}")
         except OSError as e:
             print(f"Error creating target directory {target_folder}: {e}")
             exit()


    processed_amount_file = f'results/processed_amount_{attributes["name"]}.txt'
    processed_list_file = f"results/processed_list_{attributes['name']}.txt"
    downloaded = 0
    try:
        # Initialize amount file if it doesn't exist
        if not os.path.exists(processed_amount_file):
             with open(processed_amount_file, 'w') as f:
                 f.write('0')
             print(f"Created {processed_amount_file}, starting count from 0.")
        else:
            with open(processed_amount_file) as f:
                content = f.readline()
                if content:
                     downloaded = int(content.strip())
                else: # Handle empty file case
                     downloaded = 0
                     print(f"Warning: {processed_amount_file} was empty. Starting count from 0.")
                     # Optionally rewrite '0' to the file here for consistency
                     # with open(processed_amount_file, 'w') as f_write:
                     #     f_write.write('0')
    except (IOError, ValueError) as e:
        print(f"Error reading {processed_amount_file}: {e}. Assuming 0 downloaded.")
        downloaded = 0
        # Try to reset the file to '0' if it was corrupted
        try:
            with open(processed_amount_file, 'w') as f:
                 f.write('0')
        except IOError:
            print(f"Could not even write to {processed_amount_file}. Check permissions/disk space.")
            exit()


    print(f"Already processed: {downloaded}")
    images = get_download_list(downloaded)

    if not images:
        print("No images to download based on the calculated list.")
    else:
        print(f"Attempting to download {len(images)} images...")

    successful_downloads_in_session = 0
    for i, image_id in enumerate(images):
        print("-" * 20)
        print(f"Processing image {i+1}/{len(images)}: {image_id}")
        suffix = get_suffix(image_id) # Moved suffix determination here
        if download_image(image_id, suffix):
            print(f"SUCCESS: {image_id}.{suffix} processed!")
            successful_downloads_in_session += 1
            # Log successful download immediately to processed list
            try:
                with open(processed_list_file, 'a') as f_list:
                    f_list.write(f"{image_id}\n")
            except IOError as e:
                 print(f"Error writing to {processed_list_file}: {e}")

            # Update total processed amount
            current_total_downloaded = downloaded + successful_downloads_in_session
            try:
                with open(processed_amount_file, 'w') as f_amount:
                    f_amount.write(str(current_total_downloaded))
            except IOError as e:
                print(f"CRITICAL ERROR: Could not update {processed_amount_file} to {current_total_downloaded}: {e}")
                # Decide how to handle - maybe stop the script?
                print("Stopping script to prevent count mismatch.")
                exit()
        else:
            print(f"FAILURE/SKIP: {image_id}.{suffix or '(unknown suffix)'} was not downloaded.")
            # Failure/skip is already logged within download_image or get_suffix

        # Add a small delay between images to be polite
        time.sleep(1) # 1 second delay

    print("-" * 20)
    print(f"Download session finished. Successfully downloaded {successful_downloads_in_session} new images.")
    print(f"Total processed count is now: {downloaded + successful_downloads_in_session}")