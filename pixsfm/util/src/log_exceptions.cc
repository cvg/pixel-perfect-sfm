//
// Created by dawars on 2024.03.20..
//
#include <sstream>

#include <colmap/sensor/models.h>
#include "log_exceptions.h"

std::ostream &operator<<(std::ostream &os, const colmap::CameraModelId &id) {
    os << static_cast<int>(id);
//    switch (id) {
//        case colmap::CameraModelId::kInvalid:
//            os << "kInvalid";
//            break;
//        case colmap::CameraModelId::kSimplePinhole:
//            os << "kSimplePinhole";
//            break;
//        case colmap::CameraModelId::kPinhole:
//            os << "kPinhole";
//            break;
//        case colmap::CameraModelId::kSimpleRadial:
//            os << "kSimpleRadial";
//            break;
//        case colmap::CameraModelId::kRadial:
//            os << "kRadial";
//            break;
//        case colmap::CameraModelId::kOpenCV:
//            os << "kOpenCV";
//            break;
//        case colmap::CameraModelId::kOpenCVFisheye:
//            os << "kOpenCVFisheye";
//            break;
//        case colmap::CameraModelId::kFullOpenCV:
//            os << "kFullOpenCV";
//            break;
//        case colmap::CameraModelId::kFOV:
//            os << "kFOV";
//            break;
//        case colmap::CameraModelId::kSimpleRadialFisheye:
//            os << "kSimpleRadialFisheye";
//            break;
//        case colmap::CameraModelId::kRadialFisheye:
//            os << "kRadialFisheye";
//            break;
//        case colmap::CameraModelId::kThinPrismFisheye:
//            os << "kThinPrismFisheye";
//            break;
//    }
    return os;
}
