"""Title folding, shared by every script that matches a show's name across sources."""

import re
import unicodedata


def fold(title: str) -> str:
    """Lowercase and strip accents and punctuation: 'Les Misérables' and 'Les Miserables' both -> 'les miserables'."""
    # Apostrophes go before anything else: "Band's" and "Band’s" must both become "bands", not "band s"
    # and "bands". '&' is written out because MusicBrainz has 'Love & Murder' where others have 'and'.
    t = re.sub(r"['’‘`]", "", title).replace("&", " and ")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    # Grosses often add a suffix other sources leave off ('Billy Elliot: The Musical', 'Moulin Rouge! The Musical!')
    return re.sub(r" the musical$", "", t)
