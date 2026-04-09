//core/math/math_defs.h

#ifndef MATH_DEFS_H
#define MATH_DEFS_H

#include "src/big_number.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * @typedef real_t
 * @brief The fundamental numeric type for ETengine.
 * 
 * By using BigNumber as the base real type, we ensure that every position, 
 * velocity, and physical property is represented with arbitrary precision 
 * and fixed-point determinism, eliminating floating-point non-determinism.
 */
typedef BigNumber real_t;

enum Vector2Axis {
	AXIS_X,
	AXIS_Y,
};

enum Vector3Axis {
	AXIS_V3_X,
	AXIS_V3_Y,
	AXIS_V3_Z,
};

enum Vector4Axis {
	AXIS_V4_X,
	AXIS_V4_Y,
	AXIS_V4_Z,
	AXIS_V4_W,
};

enum EulerOrder {
	EULER_ORDER_XYZ,
	EULER_ORDER_XZY,
	EULER_ORDER_YXZ,
	EULER_ORDER_YZX,
	EULER_ORDER_ZXY,
	EULER_ORDER_ZYX,
};

enum ClockDirection {
	CLOCKWISE,
	COUNTERCLOCKWISE
};

enum Orientation {
	HORIZONTAL,
	VERTICAL
};

enum Side {
	SIDE_LEFT,
	SIDE_TOP,
	SIDE_RIGHT,
	SIDE_BOTTOM,
};

enum Corner {
	CORNER_TOP_LEFT,
	CORNER_TOP_RIGHT,
	CORNER_BOTTOM_RIGHT,
	CORNER_BOTTOM_LEFT,
};

// Math Macro Utilities for BigNumber
#define SIGN(m_v) ((m_v) < BigNumber(0) ? BigNumber(-1) : ((m_v) > BigNumber(0) ? BigNumber(1) : BigNumber(0)))
#define ABS(m_v) BigNumber::abs(m_v)
#define SQR(m_v) ((m_v) * (m_v))

// ETengine specific bit-perfect epsilon for galactic scale
#define BIG_EPSILON BigNumber("0.00000000000001")

#endif // MATH_DEFS_H