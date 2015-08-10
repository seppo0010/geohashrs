/*
* Copyright (c) 2013-2014, yinqiwen <yinqiwen@gmail.com>
* Copyright (c) 2014, Matt Stancliff <matt@genges.com>.
* Copyright (c) 2015, Salvatore Sanfilippo <antirez@gmail.com>.
* Copyright (c) 2015, Sebastian Waisbrot <seppo0010@gmail.com>.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of Redis nor the names of its contributors may be used
*    to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
*/

use std::cmp::Ordering;
use std::f64::consts::PI;

pub const STEP_MAX: usize = 26;

/// Limits from EPSG:900913 / EPSG:3785 / OSGEO:41001
pub const LAT_MIN:f64 = -85.05112878;
pub const LAT_MAX:f64 = 85.05112878;
pub const LONG_MIN:f64 = -180.0;
pub const LONG_MAX:f64 = 180.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Bits {
    bits: u64,
    step: u8,
}

impl Bits {
    pub fn new(bits: u64, step: u8) -> Self {
        Bits { bits: bits, step: step }
    }

    fn clear(&mut self) {
        self.bits = 0;
        self.step = 0;
    }

    fn is_zero(&self) -> bool {
        self.bits == 0 || self.step == 0
    }

    pub fn decode(&self) -> Option<Area> {
        RangeLatLon::new().decode(self.clone())
    }

    pub fn decode_to_long_lat(&self) -> Option<(f64, f64)> {
        self.decode().map(|ha| ha.decode_to_long_lat())
    }

    fn move_x(&mut self, d: i8) {
        if d == 0 {
            return;
        }

        let mut x = self.bits & 0xaaaaaaaaaaaaaaaa;
        let y = self.bits & 0x5555555555555555;
        let zz = 0x5555555555555555 >> (64 - self.step * 2);

        if d > 0 {
            x += zz + 1;
        } else {
            x |= zz;
            x -= zz + 1;
        }

        x &= 0xaaaaaaaaaaaaaaaa >> (64 - self.step * 2);
        self.bits = x | y;
    }

    fn move_y(&mut self, d: i8) {
        if d == 0 {
            return;
        }

        let x = self.bits & 0xaaaaaaaaaaaaaaaa;
        let mut y = self.bits & 0x5555555555555555;

        let zz = 0xaaaaaaaaaaaaaaaa >> (64 - self.step * 2);

        if d > 0 {
            y += zz + 1;
        } else {
            y |= zz;
            y -= zz + 1;
        }
        y &= 0x5555555555555555 >> (64 - self.step * 2);
        self.bits = x | y;
    }
}

impl PartialOrd for Bits {
    fn partial_cmp(&self, other: &Bits) -> Option<Ordering> {
        if self.step != other.step {
            self.step.partial_cmp(&other.step)
        } else {
            self.bits.partial_cmp(&other.bits)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    min: f64,
    max: f64,
}

impl Range {
    fn is_zero(&self) -> bool {
        self.min == 0.0 || self.max == 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RangeLatLon {
    long: Range,
    lat: Range,
}

impl RangeLatLon {
    pub fn new() -> Self {
        RangeLatLon {
            long: Range {max: LONG_MAX, min: LONG_MIN },
            lat: Range {max: LAT_MAX, min: LAT_MIN },
        }
    }

    // TODO: better error handling
    pub fn encode(&self, longitude: f64, latitude: f64, step: u8) -> Option<Bits> {
        // Check basic arguments sanity
        if step > 32 || step == 0 || self.lat.is_zero() || self.long.is_zero() {
            return None;
        }

        // check constraints
        if longitude > LONG_MAX || longitude < LONG_MIN || latitude > LAT_MAX || latitude < LAT_MIN {
            return None;
        }

        if latitude < self.lat.min || latitude > self.lat.max || longitude < self.long.min || longitude > self.long.max {
            return None;
        }

        let mut lat_offset = (latitude - self.lat.min) / (self.lat.max - self.lat.min);
        let mut long_offset = (longitude - self.long.min) / (self.long.max - self.long.min);

        // convert to fixed point based on the step size
        lat_offset *= (1 << step) as f64;
        long_offset *= (1 << step) as f64;

        let bits = interleave64(lat_offset as u32, long_offset as u32);

        return Some(Bits::new(bits, step));
    }

    // TODO: better error handling
    pub fn decode(&self, hash: Bits) -> Option<Area> {
        if hash.is_zero() || self.lat.is_zero() || self.long.is_zero() {
            return None;
        }

        let step = hash.step;
        let hash_sep = deinterleave64(hash.bits); // hash = [LAT][LONG]

        let lat_scale = self.lat.max - self.lat.min;
        let long_scale = self.long.max - self.long.min;

        let ilato = hash_sep as u32 as f64; // get lat part of deinterleaved hash
        let ilono = (hash_sep >> 32) as u32 as f64; // shift over to get long part of hash

        // divide by 2**step
        // Then, for 0-1 coordinate, multiply times scale and add
        // to the min to get the absolute coordinate

        let step_shift = (1 << step) as f64;

        Some(Area::new(hash,
                    Range {
                        max: self.long.min + ((ilono + 1.0) * 1.0 / step_shift) * long_scale,
                        min: self.long.min + (ilono * 1.0 / step_shift) * long_scale
                    },
                    Range {
                        max: self.lat.min + (ilato * 1.0 / step_shift) * lat_scale,
                        min: self.lat.min + ((ilato + 1.0) * 1.0 / step_shift) * lat_scale
                    }
                    ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Area {
    hash: Bits,
    longitude: Range,
    latitude: Range,
}

impl Area {
    pub fn new(hash: Bits, longitude: Range, latitude: Range) -> Self {
        Area { hash: hash, longitude: longitude, latitude: latitude }
    }

    pub fn decode_to_long_lat(&self) -> (f64, f64) {
        (
        (self.longitude.min + self.longitude.max) / 2.0,
        (self.latitude.min + self.latitude.max) / 2.0,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Neighbors {
    north: Bits,
    east: Bits,
    west: Bits,
    south: Bits,
    south_east: Bits,
    south_west: Bits,
    north_east: Bits,
    north_west: Bits,
}

impl Neighbors {
    pub fn new(hash: &Bits) -> Self {
        let mut this = Neighbors {
            north: hash.clone(),
            east: hash.clone(),
            west: hash.clone(),
            south: hash.clone(),
            south_east: hash.clone(),
            south_west: hash.clone(),
            north_east: hash.clone(),
            north_west: hash.clone(),
        };

        this.east.move_x(1);
        this.east.move_y(0);

        this.west.move_x(-1);
        this.west.move_y(0);

        this.south.move_x(0);
        this.south.move_y(-1);

        this.north.move_x(0);
        this.north.move_y(1);

        this.north_west.move_x(-1);
        this.north_west.move_y(1);

        this.north_east.move_x(1);
        this.north_east.move_y(1);

        this.south_east.move_x(1);
        this.south_east.move_y(-1);

        this.south_west.move_x(-1);
        this.south_west.move_y(-1);

        return this;
    }
}

/// ing works like this:
/// Divide the world into 4 buckets.  Label each one as such:
///  -----------------
///  |       |       |
///  |       |       |
///  | 0,1   | 1,1   |
///  -----------------
///  |       |       |
///  |       |       |
///  | 0,0   | 1,0   |
///  -----------------

/// Interleave lower bits of x and y, so the bits of x
/// are in the even positions and bits from y in the odd;
/// x and y must initially be less than 2**32 (65536).
/// From:  https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN

#[inline]
fn interleave64(xlo: u32, ylo: u32) -> u64 {
    const B:[u64; 5] = [0x5555555555555555, 0x3333333333333333,
        0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF,
        0x0000FFFF0000FFFF];
    const S:[u64; 5] = [1, 2, 4, 8, 16];

    let mut x = xlo as u64;
    let mut y = ylo as u64;

    x = (x | (x << S[4])) & B[4];
    y = (y | (y << S[4])) & B[4];

    x = (x | (x << S[3])) & B[3];
    y = (y | (y << S[3])) & B[3];

    x = (x | (x << S[2])) & B[2];
    y = (y | (y << S[2])) & B[2];

    x = (x | (x << S[1])) & B[1];
    y = (y | (y << S[1])) & B[1];

    x = (x | (x << S[0])) & B[0];
    y = (y | (y << S[0])) & B[0];

    return x | (y << 1);
}

/// Eeverse the interleave process
/// Derived from http://stackoverflow.com/questions/4909263
fn deinterleave64(interleaved: u64) -> u64 {
    const B:[u64; 6] = [0x5555555555555555, 0x3333333333333333,
        0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF,
        0x0000FFFF0000FFFF, 0x00000000FFFFFFFF];
    const S:[u64; 6] = [0, 1, 2, 4, 8, 16];

    let mut x = interleaved;
    let mut y = interleaved >> 1;

    x = (x | (x >> S[0])) & B[0];
    y = (y | (y >> S[0])) & B[0];

    x = (x | (x >> S[1])) & B[1];
    y = (y | (y >> S[1])) & B[1];

    x = (x | (x >> S[2])) & B[2];
    y = (y | (y >> S[2])) & B[2];

    x = (x | (x >> S[3])) & B[3];
    y = (y | (y >> S[3])) & B[3];

    x = (x | (x >> S[4])) & B[4];
    y = (y | (y >> S[4])) & B[4];

    x = (x | (x >> S[5])) & B[5];
    y = (y | (y >> S[5])) & B[5];

    return x | (y << 32);
}


// helper

const D_R:f64 = PI / 180.0;

/// The usual PI/180 constant
const DEG_TO_RAD:f64 = 0.017453292519943295769236907684886;
/// Earth's quatratic mean radius for WGS-84
const EARTH_RADIUS_IN_METERS:f64 = 6372797.560856;

const MERCATOR_MAX:f64 = 20037726.37;

#[inline]
fn deg_rad(ang: f64) -> f64 { ang * D_R }
fn rad_deg(ang: f64) -> f64 { ang / D_R }

pub struct Radius {
    area: Area,
    neighbors: Neighbors,
}

impl Radius {
    pub fn new(area: Area, neighbors: Neighbors) -> Self {
        Radius { area: area, neighbors: neighbors }
    }
}

/// You must *ONLY* estimate steps when you are encoding.
/// If you are decoding, always decode to GEO_STEP_MAX (26)
fn estimate_steps_by_radius(mut range_meters: f64, lat: f64) -> u8 {
    if range_meters == 0.0 {
        return 26;
    }
    let mut step = 1;
    while range_meters < MERCATOR_MAX {
        range_meters *= 2.0;
        step += 1;
    }

    step -= 2; // Make sure range is included in the worst case

    // Wider range torwards the poles... Note: it is possible to do better
    // than this approximation by computing the distance between meridians
    // at this latitude, but this does the trick for now

    if lat > 67.0 || lat < -67.0 { step -= 1; }
    if lat > 80.0 || lat < -80.0 { step -= 1; }

    // Frame to valid range
    if step < 1 { step = 1; }
    if step > 26 { step = 25; }

    return step;
}

fn bounding_box(longitude: f64, latitude: f64, radius_meters: f64) -> (f64, f64, f64, f64) {
    let lonr = deg_rad(longitude);
    let latr = deg_rad(latitude);

    let distance = radius_meters / EARTH_RADIUS_IN_METERS;
    let min_latitude = latr - distance;
    let max_latitude = latr + distance;

    let difference_longitude = (distance.sin() / latr.cos()).asin();
    let min_longitude = lonr - difference_longitude;
    let max_longitude = lonr + difference_longitude;

    (
     rad_deg(min_longitude),
     rad_deg(min_latitude),
     rad_deg(max_longitude),
     rad_deg(max_latitude),
    )
}

pub fn get_areas_by_radius(longitude: f64, latitude: f64, radius_meters: f64) -> Radius {
    let (min_lon, min_lat, max_lon, max_lat) = bounding_box(longitude, latitude, radius_meters);
    let steps = estimate_steps_by_radius(radius_meters, latitude);

    let range = RangeLatLon::new();
    let hash = range.encode(longitude, latitude, steps).unwrap();
    let mut neighbors = Neighbors::new(&hash);
    let range = RangeLatLon::new();
    let area = range.decode(hash).unwrap();

    if area.latitude.min < min_lat {
        neighbors.south.clear();
        neighbors.south_west.clear();
        neighbors.south_east.clear();
    }
    if area.latitude.max > max_lat {
        neighbors.north.clear();
        neighbors.north_east.clear();
        neighbors.north_west.clear();
    }
    if area.longitude.min < min_lon {
        neighbors.west.clear();
        neighbors.south_west.clear();
        neighbors.north_west.clear();
    }
    if area.longitude.max > max_lon {
        neighbors.east.clear();
        neighbors.south_east.clear();
        neighbors.north_east.clear();
    }
    Radius::new(area, neighbors)
}

/// Calculate distance using haversin great circle distance formula
pub fn get_distance(lon1d: f64, lat1d: f64, lon2d: f64, lat2d: f64) -> f64 {
    let lat1r = deg_rad(lat1d);
    let lon1r = deg_rad(lon1d);
    let lat2r = deg_rad(lat2d);
    let lon2r = deg_rad(lon2d);
    let u = ((lat2r - lat1r) / 2.0).sin();
    let v = ((lon2r - lon1r) / 2.0).sin();
    2.0 * EARTH_RADIUS_IN_METERS * (u * u + lat1r.cos() * lat2r.cos() * v * v).sqrt().asin()
}

pub fn get_distance_if_in_radius(x1: f64, y1: f64, x2: f64, y2: f64, radius: f64) -> Option<f64> {
    let distance = get_distance(x1, y1, x2, y2);
    if distance > radius {
        None
    } else {
        Some(distance)
    }

}
