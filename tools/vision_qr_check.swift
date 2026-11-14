import Foundation
import Vision
import CoreVideo
import CoreGraphics
import ImageIO

struct QRResult: Codable {
    let path: String
    let vision: [String]
}

func make420fPixelBuffer(from cgImage: CGImage) -> CVPixelBuffer? {
    let width = cgImage.width
    let height = cgImage.height
    let w = width % 2 == 0 ? width : width + 1
    let h = height % 2 == 0 ? height : height + 1

    let bytesPerRow = w * 4
    var rgba = [UInt8](repeating: 255, count: bytesPerRow * h)
    guard let ctx = CGContext(
        data: &rgba,
        width: w,
        height: h,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
        return nil
    }

    ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.draw(cgImage, in: CGRect(x: 0, y: h - height, width: width, height: height))

    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        w,
        h,
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        nil,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let pb = pixelBuffer else {
        return nil
    }

    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }

    guard let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0)?.assumingMemoryBound(to: UInt8.self),
          let uvBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1)?.assumingMemoryBound(to: UInt8.self) else {
        return nil
    }
    let yStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
    let uvStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)

    for y in 0..<h {
        let rowOff = y * bytesPerRow
        let yRow = y * yStride
        for x in 0..<w {
            let p = rowOff + x * 4
            let r = Int(rgba[p])
            let g = Int(rgba[p + 1])
            let b = Int(rgba[p + 2])
            let yVal = (77 * r + 150 * g + 29 * b + 128) >> 8
            yBase[yRow + x] = UInt8(clamping: yVal)
        }
    }

    for y in stride(from: 0, to: h, by: 2) {
        let uvRow = (y / 2) * uvStride
        let rowOff = y * bytesPerRow
        for x in stride(from: 0, to: w, by: 2) {
            let p = rowOff + x * 4
            let r = Int(rgba[p])
            let g = Int(rgba[p + 1])
            let b = Int(rgba[p + 2])
            let cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128
            let cr = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128
            uvBase[uvRow + x] = UInt8(clamping: cb)
            uvBase[uvRow + x + 1] = UInt8(clamping: cr)
        }
    }

    return pb
}

func decodeWithVision(cgImage: CGImage) throws -> [String] {
    guard let pb = make420fPixelBuffer(from: cgImage) else {
        throw NSError(domain: "VisionQRCheck", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to build 420f pixel buffer"])
    }
    var results: [String] = []
    let request = VNDetectBarcodesRequest()
    request.revision = VNDetectBarcodesRequestRevision1
    request.symbologies = [.qr]
    let handler = VNImageRequestHandler(cvPixelBuffer: pb, options: [:])
    try handler.perform([request])
    if let observations = request.results {
        for obs in observations {
            if let payload = obs.payloadStringValue {
                results.append(payload)
            }
        }
    }
    return results
}

func main() {
    let args = Array(CommandLine.arguments.dropFirst())
    if args.isEmpty {
        fputs("Usage: vision_qr_check <image_path> [<image_path> ...]\n", stderr)
        exit(2)
    }

    var outputs: [QRResult] = []
    for path in args {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            fputs("File not found: \(path)\n", stderr)
            exit(1)
        }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            fputs("Failed to load image: \(path)\n", stderr)
            exit(1)
        }
        do {
            let vResults = try decodeWithVision(cgImage: cgImage)
            outputs.append(QRResult(path: path, vision: vResults))
        } catch {
            fputs("Vision error on \(path): \(error)\n", stderr)
            exit(1)
        }
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    if outputs.count == 1 {
        let data = try! encoder.encode(outputs[0])
        print(String(data: data, encoding: .utf8)!)
    } else {
        let data = try! encoder.encode(outputs)
        print(String(data: data, encoding: .utf8)!)
    }
}

main()
