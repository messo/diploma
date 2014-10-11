/*
 * Vector.h
 *
 *  Created on: 2014.03.20.
 *      Author: Balint
 */

#ifndef VECTOR_H_
#define VECTOR_H_

namespace geom {

    struct Vector {
        float x, y, z;
        bool zero;

        Vector() {
            x = y = z = 0;
            zero = true;
        }

        Vector(float x0, float y0, float z0 = 0) {
            x = x0;
            y = y0;
            z = z0;
            zero = false;
        }

        Vector operator*(float a) const {
            return Vector(x * a, y * a, z * a);
        }

        Vector &operator*=(float a) {
            x *= a;
            y *= a;
            z *= a;
            return *this;
        }

        Vector operator/(float a) const {
            return Vector(x / a, y / a, z / a);
        }

        Vector operator+(const Vector &v) const {
            return Vector(x + v.x, y + v.y, z + v.z);
        }

        Vector &operator+=(const Vector &v) {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        Vector &operator-=(const Vector &v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        Vector operator-(const Vector &v) const {
            return Vector(x - v.x, y - v.y, z - v.z);
        }

        float operator*(const Vector &v) const {
            return (x * v.x + y * v.y + z * v.z);
        }

        Vector operator%(const Vector &c) const {
            const Vector &b = *this;

            return Vector(b.y * c.z - b.z * c.y, b.z * c.x - b.x * c.z,
                    b.x * c.y - b.y * c.x);
        }

        Vector normal() const {
            return (*this) / length();
        }

        float length() const {
            return sqrtf(x * x + y * y + z * z);
        }

        bool isNull() const {
            return zero;
        }
    };

} /* namespace geom */

#endif /* VECTOR_H_ */
