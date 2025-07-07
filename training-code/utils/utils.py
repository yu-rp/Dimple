import hashlib

def stable_hash(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:8] 