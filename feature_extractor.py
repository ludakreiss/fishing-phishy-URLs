from urllib.parse import urlparse
import ipaddress
import re
import pandas as pd
import whois
import dns.resolver
import dns.rdatatype
import dns.exception
import streamlit as st
from datetime import date 
import requests
from serpapi import GoogleSearch
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
    with open("Processed Data/shortening_services.txt", 'r') as f:
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

def registrationLength(url):
    """
    This function checks how long the domain has to expire for.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 1 if domain expires in a year or less,
         otherwise returns 0.
    """
    domain = urlparse(url).netloc

    try:
        domain = whois.whois(domain)
    except:
        return 1

    expiration_date = domain.expiration_date
    current_date = date.today()

    if expiration_date is None:
        return 1


    if isinstance(expiration_date, list):
        for expire in expiration_date:
            if expire is not None:
                expiration_date = expire
                break
        else:
            return 1

    if hasattr(expiration_date, "date"):
        expiration_date = expiration_date.date()

    days_left = (expiration_date - current_date).days

    return 1 if days_left < 365 else 0


def getDNSRecord(url):
    """
    This function checks if there is a record for the domain in the DNS.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 1 if no record in DNS,
         otherwise returns 0.
    """
    
    domain = urlparse(url).netloc

    if re.match(r'^www\.', domain):
        domain = domain.removeprefix("www.")


    answers = []

    for query_type in dns.rdatatype.RdataType:
        try:
            answers.extend(list(dns.resolver.resolve(domain, query_type)))
        except dns.exception.DNSException:
            continue
    
    return 0 if answers else 1



def getWHOISDomain(url):
    """
    This function attempts to perform a WHOIS lookup on the given URL's domain.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 0 if WHOIS information was successfully retrieved,
         otherwise returns 1.
    """
    domain = urlparse(url).netloc

    try:
        domain = whois.whois(domain)
        return 0
    except:
        return 1


def isGoogleIndex(url):
    """
    This function checks if URL is indexed by Google using SerpAPI.

    Parameters:
    -----------
        URL (str): The URL to check.

    Returns:
    --------
        int: Returns 0 if the domain is found in Google's search results,
        otherwise returns 1.
    """
    api_key = st.secrets["SERPAPI_API_KEY"]
    domain = urlparse(url).netloc

    params = {
        "engine": "google",
        "q": f"site:{domain}",
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic_results = results.get("organic_results", [])

    for result in organic_results:
        if url in result.get("link", ""):
            return 0

    return 1


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
        'IP_Address': [int(isIPInDomain(url))],
        'Prefix/Suffix_in_Domain': [int(isPrefixSuffixInDomain(url))],
        'Tiny_URL': [int(isTinyUrl(url))],
        '@_Symbol': [int(containsAtSymbol(url))],
        'URL_Length': [getUrlLength(url)],
        'Http/https_in_Domain': [int(isProtocolInDomain(url))],
        'Depth_Of_URL': [depthOfUrl(url)],
        'Redirection': [int(redirection(url))],
        'Google_Index': [int(isGoogleIndex(url))],
        'WHOIS_Domain': [int(getWHOISDomain(url))],
        'DNS_Record': [int(getDNSRecord(url))],
        'Registration_Length': [int(registrationLength(url))],
        'Num_of_Dots': [url.count('.')],
        'Num_of_Hyphens': [url.count('-')],
        'Num_of_Underscore':[ url.count('_')]
    })
