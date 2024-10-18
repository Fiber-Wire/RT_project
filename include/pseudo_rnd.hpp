//
// Created by JCW on 2024/10/18.
//

#ifndef PSEUDO_RND_HPP
#define PSEUDO_RND_HPP

__host__ inline float get_rnd(const int id) {
    constexpr float rnd[64] = {
        0.323370, 0.269726, 0.208229, 0.084870, 0.384229, 0.754794, 0.989512, 0.115898, 0.676739, 0.761827, 0.708466, 0.481340,
        0.304094, 0.987709, 0.875270, 0.850630, 0.637664, 0.249852, 0.211666, 0.413103, 0.604806, 0.667506, 0.709824, 0.325835,
        0.890805, 0.146125, 0.760791, 0.819518, 0.386267, 0.962533, 0.095690, 0.478487, 0.221509, 0.382386, 0.343701, 0.435379,
        0.381173, 0.461499, 0.251420, 0.775184, 0.010580, 0.486213, 0.564840, 0.355930, 0.816577, 0.776045, 0.196225, 0.713191,
        0.137972, 0.392296, 0.006581, 0.903072, 0.447164, 0.815050, 0.284912, 0.817340, 0.632427, 0.191933, 0.068535, 0.828934,
        0.631082, 0.034388, 0.357435, 0.677837
    };
    return rnd[id % 64];
}
#endif //PSEUDO_RND_HPP
