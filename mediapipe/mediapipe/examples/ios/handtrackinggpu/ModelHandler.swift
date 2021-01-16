//
//  File.swift
//  Mediapipe
//
//  Created by  dmitry on 04.01.2021.
//

import Foundation
import TensorFlowLite


@objc public class ModelHandler: NSObject {
    
    var result: [Float]

    @objc public override init() {
        result = [Float]()
    }
    
    @objc public func predict(_ nsdata: NSData) -> Int {
        guard
          let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite")
        else {
            return -1
        }

        do {
            let interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
            let inputTensor = try interpreter.input(at: 0)
            var data = nsdata as Data
            
            try interpreter.copy(data, toInputAt: 0)
            try interpreter.invoke()
            let outputTensor = try interpreter.output(at: 0)
         
            result =  [Float](unsafeData: outputTensor.data) ?? []
            print("PREDICT: \(result)")
    
        } catch {
            print(error.localizedDescription)
        }
        print("ARGMAX: \(result.argmax()!)")
        return result.argmax()!
    }
}


extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
    
}

extension Array where Element: Comparable {
    func argmax() -> Index? {
        return indices.max(by: { self[$0] < self[$1] })
    }
}
