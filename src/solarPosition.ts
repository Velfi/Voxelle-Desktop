/**
 * Simplified NOAA solar position algorithm.
 * Accurate to ~0.5 degrees — more than sufficient for lighting.
 */

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

export function getSunPosition(
  date: Date,
  latitudeDeg: number,
  longitudeDeg: number,
): { azimuthDeg: number; altitudeDeg: number } {
  // Julian date from Unix timestamp (UTC).
  const jd = 2440587.5 + date.getTime() / 86_400_000;
  const T = (jd - 2451545.0) / 36525.0; // Julian century

  // --- Solar geometry ---
  const L0 = mod360(280.46646 + 36000.76983 * T); // mean longitude
  const M = mod360(357.52911 + 35999.05029 * T); // mean anomaly (deg)
  const Mrad = M * DEG;

  // Equation of center
  const C =
    (1.914602 - 0.004817 * T) * Math.sin(Mrad) +
    0.019993 * Math.sin(2 * Mrad) +
    0.000289 * Math.sin(3 * Mrad);

  const sunLon = L0 + C; // sun true longitude (deg)
  const omega = 125.04 - 1934.136 * T;
  const apparentLon = sunLon - 0.00569 - 0.00478 * Math.sin(omega * DEG);

  // Obliquity of the ecliptic
  const obliquity = 23.439291 - 0.0130042 * T + 0.00256 * Math.cos(omega * DEG);
  const oblRad = obliquity * DEG;

  // Sun declination
  const sinDec = Math.sin(oblRad) * Math.sin(apparentLon * DEG);
  const dec = Math.asin(sinDec); // radians

  // Equation of time (minutes)
  const y = Math.tan(oblRad / 2) ** 2;
  const L0rad = L0 * DEG;
  const ecc = 0.016708634 - 0.000042037 * T;
  const eot =
    4 *
    RAD *
    (y * Math.sin(2 * L0rad) -
      2 * ecc * Math.sin(Mrad) +
      4 * ecc * y * Math.sin(Mrad) * Math.cos(2 * L0rad) -
      0.5 * y * y * Math.sin(4 * L0rad) -
      1.25 * ecc * ecc * Math.sin(2 * Mrad));

  // True solar time (minutes from midnight UTC) adjusted for longitude.
  const utcMins = ((jd + 0.5) % 1) * 1440;
  const trueSolarTime = mod(utcMins + eot + 4 * longitudeDeg, 1440);

  // Hour angle (degrees, negative = before solar noon).
  const ha = (trueSolarTime / 4 - 180) * DEG;

  // --- Horizontal coordinates ---
  const lat = latitudeDeg * DEG;
  const sinAlt = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(ha);
  const altitudeDeg = Math.asin(clamp(sinAlt, -1, 1)) * RAD;

  // Azimuth (CW from North, 0-360).
  const cosAlt = Math.cos(altitudeDeg * DEG);
  let azimuthDeg: number;
  if (cosAlt === 0) {
    azimuthDeg = 0;
  } else {
    const cosAz = (Math.sin(dec) - Math.sin(lat) * sinAlt) / (Math.cos(lat) * cosAlt);
    azimuthDeg = Math.acos(clamp(cosAz, -1, 1)) * RAD;
    if (ha > 0) azimuthDeg = 360 - azimuthDeg;
  }

  return { azimuthDeg, altitudeDeg };
}

function mod360(v: number): number {
  return ((v % 360) + 360) % 360;
}

function mod(v: number, m: number): number {
  return ((v % m) + m) % m;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
