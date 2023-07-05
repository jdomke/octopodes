#ifndef RAND_GEN_H
#define RAND_GEN_H

#include <random>
#include <type_traits>
#include <limits>
#include <mkl.h>

using namespace std;
                  //below code segment taken from common_func.c which comes with the installation of Intel oneAPI Math Kernel Library for use with cblas_hgemm() function
typedef union {       
  MKL_F16 raw;
  struct {
    unsigned int frac : 10;
    unsigned int exp  :  5;
    unsigned int sign :  1;
  } bits;
} conv_union_f16; 

typedef union {
  float raw;
  struct {
    unsigned int frac : 23;
    unsigned int exp  :  8;
    unsigned int sign :  1;
  } bits;
} conv_union_f32;

static float h2f(MKL_F16 x) {       //convert half precision to f16, taken from common_func.c
  conv_union_f16 src;
  conv_union_f32 dst;

  src.raw = x;
  dst.raw = 0;
  dst.bits.sign = src.bits.sign;

  if (src.bits.exp == 0x01f) {
    dst.bits.exp = 0xff;
    if (src.bits.frac > 0) {
      dst.bits.frac = ((src.bits.frac | 0x200) << 13);
    }
  } else if (src.bits.exp > 0x00) {
    dst.bits.exp = src.bits.exp + ((1 << 7) - (1 << 4));
    dst.bits.frac = (src.bits.frac << 13);
  } else {
    unsigned int v = (src.bits.frac << 13);

    if (v > 0) {
      dst.bits.exp = 0x71;
      while ((v & 0x800000UL) == 0) {
        dst.bits.exp--;
        v <<= 1;
      }
      dst.bits.frac = v;
    }
  }

  return dst.raw;
}

static MKL_F16 f2h(float x) { //convert f16 to half precision, taken from common_func.c
  conv_union_f32 src;
  conv_union_f16 dst;

  src.raw = x;
  dst.raw = 0;
  dst.bits.sign = src.bits.sign;

  if (src.bits.exp == 0x0ff) {
    dst.bits.exp = 0x01f;
    dst.bits.frac = (src.bits.frac >> 13);
    if (src.bits.frac > 0) {
      dst.bits.frac |= 0x200;
    }
  } else if (src.bits.exp >= 0x08f) {
    dst.bits.exp = 0x01f;
    dst.bits.frac = 0x000;
  } else if (src.bits.exp >= 0x071) {
    dst.bits.exp = src.bits.exp + ((1 << 4) - (1 << 7));
    dst.bits.frac = (src.bits.frac >> 13);
  } else if (src.bits.exp >= 0x067) {
    dst.bits.exp = 0x000;
    if (src.bits.frac > 0) {
      dst.bits.frac = (((1U << 23) | src.bits.frac) >> 14);
    } else {
      dst.bits.frac = 1;
    }
  }

  return dst.raw;
}

template <typename T>
T getRandomValue(T low, T high){
    static unsigned seed_val = 15;      //set a fixed seed for testing purposes
    static mt19937 gen(seed_val); 
    if constexpr(std::is_same_v<T, MKL_F16>){     //generate random float values and convert them to mkl_f16 using f2h
        std::uniform_real_distribution<float> dis(static_cast<float>(low), static_cast<float>(high));
        return f2h(dis(gen));
    }
    else if constexpr (std::is_integral<T>::value) {      //generate int values
        std::uniform_int_distribution<T> dis(low, high);
        return dis(gen);
    }
    else if constexpr (std::is_floating_point<T>::value) {    //generate fp values
        std::uniform_real_distribution<T> dis(low, high);
        return dis(gen);
    }
    return T(0.0);
}
#endif