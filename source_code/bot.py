import pickle
import socket
import atexit

# Global counters for statistics
stats = {
    'total_requests': 0,
    'bot_requests': 0,
    'non_bot_requests': 0
}

# Known bot suffixes used for reverse DNS validation
KNOWN_BOTS = {
    "Googlebot": [".googlebot.com"],
    "Bingbot": [".search.msn.com"],
    "AhrefsBot": [".ahrefs.com", ".ahrefs.net"],
    "YandexBot": [".yandex.ru", ".yandex.com", ".yandex.net"],
    "SemrushBot": [".semrush.com"],
    "DuckDuckBot": [".duckduckgo.com"],
    "MJ12bot": [".majestic12.co.uk"],
    "Slurp": [".crawl.yahoo.net"],
    "Applebot": [".apple.com"]
}

# Cache file paths for verified and spoofed bot IPs
VERIFIED_IP_FILE = "verified_bots.pkl"
SPOOFED_IP_FILE = "spoofed_bots.pkl"

def pickle_load(path):
    """
    Load a Python set object from a pickle file.

    If the file does not exist or is invalid/corrupted, returns an empty set instead
    of raising an exception.

    Args:
        path (str): Path to the pickle file.

    Returns:
        set: A deserialized set object, or an empty set on error.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return set()

# Load IP caches early to avoid NameError during runtime
verified_bot_ips = pickle_load(VERIFIED_IP_FILE)
spoofed_bot_ips = pickle_load(SPOOFED_IP_FILE)

def save_ip_caches():
    """
    Persist the sets of verified and spoofed bot IP addresses to disk.

    This function is automatically registered to run at program exit using `atexit`,
    ensuring DNS validation results are cached across runs to minimize repeated lookups.
    """
    with open(VERIFIED_IP_FILE, "wb") as f:
        pickle.dump(verified_bot_ips, f)
    with open(SPOOFED_IP_FILE, "wb") as f:
        pickle.dump(spoofed_bot_ips, f)

atexit.register(save_ip_caches)

def reverse_dns(ip):
    """
    Perform a reverse DNS lookup for a given IP address.

    Args:
        ip (str): The IP address to resolve.

    Returns:
        str or None: The resolved hostname if successful, otherwise None.
    """
    try:
        return socket.gethostbyaddr(ip)[0]
    except socket.herror:
        return None

def forward_dns(hostname):
    """
    Perform a forward DNS lookup for a given hostname.

    Args:
        hostname (str): The domain or hostname to resolve.

    Returns:
        str or None: The resolved IP address if successful, otherwise None.
    """
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return None

def is_valid_bot(ip, ua):
    """
    Determine if a request is from a legitimate search engine bot.

    The validation involves:
        1. Checking if the User-Agent contains a known bot identifier.
        2. Validating reverse DNS resolution ends with the bot's official domain.
        3. Verifying the resolved hostname maps back to the original IP.

    Verified IPs are cached to avoid redundant lookups.
    Spoofed IPs are similarly cached for efficiency.

    Args:
        ip (str): IP address of the requester.
        ua (str): User-Agent string from the request.

    Returns:
        bool: True if the request is from a verified bot, False otherwise.
    """
    stats['total_requests'] += 1
    
    if ip in verified_bot_ips:
        stats['bot_requests'] += 1
        return True
    if ip in spoofed_bot_ips:
        stats['non_bot_requests'] += 1
        return False

    for bot, suffixes in KNOWN_BOTS.items():
        if bot.lower() in ua.lower():
            rdns = reverse_dns(ip)
            if not rdns or not any(rdns.endswith(sfx) for sfx in suffixes):
                spoofed_bot_ips.add(ip)
                stats['non_bot_requests'] += 1
                return False
            if forward_dns(rdns) != ip:
                spoofed_bot_ips.add(ip)
                stats['non_bot_requests'] += 1
                return False
            verified_bot_ips.add(ip)
            stats['bot_requests'] += 1
            return True

    stats['non_bot_requests'] += 1
    return False  # Not a known bot
