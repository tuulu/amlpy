# This script downloads RNA microarray data from the GEO database, which due to the size of the files is not included in the repo
# It downloads two datasets: one for AML samples (GSE68833, TCGA) and one for healthy samples (GSE45878, GTEx)
# The .soft file for selecting only the blood samples from the GTEx dataset is also downloaded
# The annotation files are already included in the data folder

#------------------------------------------------------------------------------------------------
# Call libraries
import requests
from pathlib import Path
import gzip

#------------------------------------------------------------------------------------------------

def get_data(url, save_dir):
    """Downloads and extracts a tar.gz file from the given URL"""
    # Get filename from URL
    filename = url.split('/')[-1]
    print(f"Downloading: {filename}")
    
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the compressed file
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        # Extract the file
        print("Extracting...")
        with gzip.open(filename, 'rb') as f_in:
            with open(save_dir / filename.replace('.gz', ''), 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Clean up
        Path(filename).unlink()
        print("Done!")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

#------------------------------------------------------------------------------------------------

def main():
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok = True) # In case it already exists
    
    # URLs to download
    urls = [
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45878/matrix/GSE45878_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68833/matrix/GSE68833_series_matrix.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45878/soft/GSE45878_family.soft.gz"
    ]
    
    # Check if files already exist in data directory
    expected_files = [url.split('/')[-1].replace('.gz', '') for url in urls]
    existing_files = [f.name for f in data_dir.iterdir() if f.is_file()]
    
    if any(file in existing_files for file in expected_files):
        print("The data files already exist. No need to download again.")
        return

    # Download each file
    for url in urls:
        get_data(url, data_dir)
    
    print("\nAll downloads completed!")

# For use as a script
if __name__ == "__main__":
    main()

#------------------------------------------------------------------------------------------------
