from urllib.parse import urlparse
import ipaddress
import re
import pandas as pd
def getDomainName(url):
    """
    This fucntion gets the domain name of the URL.

    Parameters:
    -----------
        URL (string)

    Returns:
    --------
        domain (string): The domain part of the URL.
    """

    domain = urlparse(url).netloc

    if isinstance(domain, bytes):
        domain = domain.decode('utf-8')

    if re.match(r'^www\.', domain):
        domain = domain.removeprefix("www.")

    return domain

def isIPInDomain(url):
    """
    This function checks whether the given URL contains an IP address in the domain part.

    Parameters:
    -----------
        URL (string)

    Returns:
    --------
        int: Returns 1 if the domain part of the URL is an IP address,
             otherwise returns 0.
    """
    domain = urlparse(url).netloc

    try:
        if ipaddress.ip_address(domain):
            return 1
    except ValueError:
        return 0
    
def containsAtSymbol(url):
    """
    This fucntion checks whether the given URL contains an '@' symbol.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 1 if the '@' symbol is present in the URL,
             otherwise returns 0.
    """
    if '@' in url:
        return 1
    else:
        return 0
    
def getUrlLength(url):
    """
    This fucntion checks how long a URL is.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 0 if the length of a URL is less than 54,
             otherwise returns 1.
    """
    if len(url) < 54:
        return 0
    else:
        return 1
    
def isProtocolInDomain(url):
    """
    This fucntion checks if an "HTTP" or "HTTPS" token is used in the domain part of the URL.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 1 if "HTTP" or "HTTPS" is in the domain,
             otherwise returns 0.
    """
    domain = urlparse(url).netloc
    if "http" in domain:
        return 1
    elif "https" in domain:
        return 1
    else:
        return 0
    
def isPrefixSuffixInDomain(url):
    """
    This fucntion checks if the domain part of a URL has a dash symbol (-).

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 1 if the dash symbol is in the domain,
             otherwise returns 0.
    """
    domain = urlparse(url).netloc
    if "-" in domain:
        return 1
    else:
        return 0
    
def isTinyUrl(url):
    """
    This fucntion checks if a URL shortening service is being used.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 1 if the domain is part of a URL shortening service,
             otherwise returns 0.
    """
    with open("shortening_services.txt", 'r') as f:
        shortening_services = set(line.strip().lower() for line in f if line.strip())

    domain = urlparse(url).netloc
    if domain in shortening_services:
        return 1
    else:
        return 0
    

def redirection(url):
    """
    This function checks if the URL contains '//' beyond the protocol part.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 1 if an additional '//' is found beyond the protocol,
         otherwise returns 0.
    """
    first_idx = url.find("//")
    last_idx = url.rfind("//")
    
    if first_idx != last_idx:
        return 1
    return 0

def depthOfUrl(url):
    """
    This function checks the depth of a URL based on the number of "/".

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns the number of "/"-separated segments in the path.
    """
    segment = urlparse(url).path.split('/')
    depth = 0
    for i in range(len(segment)):
        if len(segment[i]) != 0:
            depth = depth+1
    return depth


def extract_features(url):
    """
    This function extracts the features of a URLs that are in a Dataframe.

    Parameters:
    -----------
        df (Dataframe): Dataframe with a 'url' named column which contains URLs.

    Returns:
    --------
        pd.DataFrame (Dataframe): Returns a new Dataframe that contains the features of the processed Dataframe that was passed.
    """
    
    return pd.DataFrame({
        'IP_Address': int(isIPInDomain(url)),
        'Prefix/Suffix_in_Domain': int(isPrefixSuffixInDomain(url)),
        'Tiny_URL': int(isTinyUrl(url)),
        '@_Symbol': int(containsAtSymbol(url)),
        'URL_Length': getUrlLength(url),
        'Http/https_in_Domain': int(isProtocolInDomain(url)),
        'Depth_Of_URL': depthOfUrl(url),
        'Redirection': int(redirection(url)),
        'Num_of_Dots': url.count('.'),
        'Num_of_Hyphens': url.count('-'),
        'Num_of_Underscore': url.count('_')
    })
