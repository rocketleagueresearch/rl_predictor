# Shared player-name normalisation for Rocket League replay data.
#
# Keep this file as the single source of truth for replay username aliases.
# Only add aliases when the mapping is known and verified.
#
# IMPORTANT:
# - This file normalises player display names only.
# - It must NOT be used to infer team names.
# - Do not use fuzzy matching in modelling code.


PLAYER_ALIASES = {
    # --------------------------------------------------------
    # Existing Open 1 aliases
    # --------------------------------------------------------
    "MayKinhoo": "MayKo",
    "Hyderr (new binds)": "Hyderr",
    "Radosinho": "Radosin",
    "acro": "accro",
    "saizenrl": "saizen",
    "AJG": "ajg.",
    "bhavi": "ajg.",

    # --------------------------------------------------------
    # Open 2 / Open 3 replay-name variants
    # --------------------------------------------------------
    "acro.": "accro",
    "radosin75": "Radosin",
    "radosinho": "Radosin",
    "Atow Rikow !": "Atow.",
    "AcroniK. 0": "AcroniK.",
    "$ARCHIE": "Archie",
    "ARCHIE$": "Archie",
    "ARCHIE": "Archie",
    "ExoTiiK": "ExoTiik",
    "joreuz": "Joreuz",
    "MaRc_By_8.": "MaRc_By_8",
    "Rebmob_.": "Rebmob",
    "Sharkdop.": "Shark.",
    "jellyrl_ [LIVE]": "jelly.",
    "JWeyts": "Jweyts",
}


def normalise_player_name(name):
    """
    Return a verified canonical player name.

    This performs exact alias replacement only.
    Unknown names are returned unchanged.
    """

    if name is None:
        return ""

    text = str(name).strip()

    if text == "":
        return ""

    return PLAYER_ALIASES.get(
        text,
        text,
    )
