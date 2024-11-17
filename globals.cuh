//
// Created by twhdkx3411 on 24-11-17.
//

#ifndef GLOBALS_CUH
#define GLOBALS_CUH

inline auto BLOCKDIMS = dim3(64);
/// x, y
inline auto GRIDDIMS = dim3(400 * 400/BLOCKDIMS.x);

#endif //GLOBALS_CUH
