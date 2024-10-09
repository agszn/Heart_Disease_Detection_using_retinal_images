# myapp/utils.py
import re

BLACKLISTED_DOMAINS = {
    "example-phishing.com",
    "bad-website.net",
    "malicious-site.org"
}

def is_phishing_url(url):
    """
    Check if a given URL is potentially a phishing URL.

    Parameters:
    url (str): The URL to be checked.

    Returns:
    bool: True if the URL is potentially a phishing URL, False otherwise.
    """

    # Check against blacklisted domains
    for domain in BLACKLISTED_DOMAINS:
        if domain in url:
            return True

    # Check for common phishing patterns
    phishing_patterns = [
        re.compile(r"login.*", re.IGNORECASE),
        re.compile(r"verify.*", re.IGNORECASE),
        re.compile(r"account.*", re.IGNORECASE),
        re.compile(r"update.*", re.IGNORECASE),
        re.compile(r"secure.*", re.IGNORECASE),
        re.compile(r"bank.*", re.IGNORECASE),
        re.compile(r"paypal.*", re.IGNORECASE)
    ]

    for pattern in phishing_patterns:
        if pattern.search(url):
            return True

    # Check for suspicious URL characteristics
    if len(url.split('.')) > 3:  # e.g., subdomain.subdomain.example.com
        return True

    if '@' in url:  # e.g., http://example.com@malicious.com
        return True

    return False
