import importlib.util
import sys

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"❌ MISSING: {package_name}")
        return False
    else:
        print(f"✅ INSTALLED: {package_name}")
        return True

print("Checking dependencies for NCBISearchTool...\n")

# Check Standard Libraries (Just to be safe, though these should be present)
check_package("asyncio")
check_package("xml") 

# Check Third-Party Libraries
missing = []
if not check_package("aiohttp"): missing.append("aiohttp")
if not check_package("aiofiles"): missing.append("aiofiles")
if not check_package("regex"): missing.append("regex")

print("\n--------------------------------")
if missing:
    print("Run the following command to install missing dependencies:")
    print(f"pip install {' '.join(missing)}")
else:
    print("All necessary packages are installed!")