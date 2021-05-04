
import pkgutil

# read version string defined in ./version.txt
__version__ = pkgutil.get_data(__name__, 'version.txt').decode().strip()