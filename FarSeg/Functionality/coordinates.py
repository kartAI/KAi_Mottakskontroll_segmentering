# FarSeg/Functionality/coordinates.py

# Libraries:

import numpy as np

# Functions:

"""
Functions converting between radians and degrees.

Arg:
    x (float): Angle in radians or degrees

Returns:
    float: Angle in radians or degrees
"""

def deg2rad(x):
    return x * np.pi / 180

def rad2deg(x):
    return x * 180 / np.pi

def UTMtoLatLon(x, y, zone):
    """
    Converts coordinates from UTM North (x) and East (y) to latitude and longitude.

    Args:
        x (float): North coordinate in UTM
        y (float): East coordinate in UTM
        zone (string): String describing the UTM zone
    """

    """
    System:
    EUREF89
    """
    a, b = 6378137, 6356752.3141
    e = np.sqrt((a**2 - b**2) / a**2)

    # Converting constants:
    y = y - 500000
    k0 = 0.9996
    m = x / k0
    mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    e1 = (1 - np.sqrt(1  - e**2)) / (1 + np.sqrt(1 - e**2))

    # Foot print of the latitude
    J1 = 3 * e1 / 2 - 27 * e1**3 / 32
    J2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    J3 = 151 * e1**3 / 96
    J4 = 1097 * e1**4 / 512

    fp = mu + J1 * np.sin(2 * mu) + J2 * np.sin(4 * mu) + J3 * np.sin(6 * mu) + J4 * np.sin(8 * mu)

    e2 = e**2 / (1 - e**2)
    C1 = e2 * (np.cos(fp))**2
    T1 = (np.tan(fp))**2
    R1 = a * (1 - e**2) / (1 - e**2 * (np.sin(fp))**2)**(3/2)
    N1 = a / np.sqrt(1 - e**2 * (np.sin(fp))**2)
    D = y / (N1 * k0)

    Q1 = N1 * np.tan(fp) / R1
    Q2 = D**2 / 2
    Q3 = (5 + 3 * T1 + 10 * C1 - 4 * C1**2 - 9 * e2) * D**6 / 720
    Q4 = (61 + 90*T1 + 298 * C1 + 45 * T1**2 - 3 * C1**2 - 252 * e2) * D**6 / 120
    Q5 = D
    Q6 = (1 + 2 * T1 + C1) * D**3 / 6
    Q7 = (5 - 2 * C1 + 28 * T1 - 3 * C1**2 + 8 * e2 + 24 * T1**2) * D**5 / 120

    if zone == "32N":
        l0 = 9
    elif zone == "33N":
        l0 = 15
    
    lat = rad2deg(fp - Q1 * (Q2 - Q3 + Q4))
    lon = l0 + rad2deg((Q5 - Q6 + Q7) / np.cos(fp))

    return lat, lon