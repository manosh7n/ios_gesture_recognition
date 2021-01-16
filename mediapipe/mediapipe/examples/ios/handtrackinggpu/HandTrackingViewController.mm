// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "HandTrackingViewController.h"
#include "mediapipe/framework/formats/landmark.pb.h"

static const char* kLandmarksOutputStream = "hand_landmarks";
static const char* kNumHandsInputSidePacket = "num_hands";

// Max number of hands to detect/process.
static const int kNumHands = 2;

@implementation HandTrackingViewController

#pragma mark - UIViewController methods

- (void)viewDidLoad {
    [super viewDidLoad];
    _model = [[ModelHandler alloc] init];
    _data = [[NSMutableArray alloc] init];
    _isRecording = NO;
    _labels = @[@"Привет", @"Деньги", @"Пока"];
    
    // UIButton initialize
    _button = [UIButton buttonWithType:UIButtonTypeSystem];
    [_button addTarget:self
                       action:@selector(recordButtonClicked:)
                       forControlEvents:UIControlEventTouchUpInside];
    [_button setTitle:@"Start" forState:UIControlStateNormal];
    _button.frame = CGRectMake(self.view.frame.size.width/2 - 45, 470, 90, 90);
//    [_button setExclusiveTouch:YES];
    _button.layer.cornerRadius = 45;
    _button.layer.borderColor = [UIColor whiteColor].CGColor;
    [_button setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    _button.layer.borderWidth = 1.0f;
    [self.view addSubview:_button];
    
    // Circle predict
    _circleView = [[UIView alloc] initWithFrame:CGRectMake(10,25,110,35)];
    _circleView.alpha = 0.65;
    _circleView.layer.cornerRadius = 15;
    _circleView.layer.borderColor = [UIColor whiteColor].CGColor;
    _circleView.layer.borderWidth = 2.0f;
    [self.view addSubview:_circleView];
    
    // UILabel predict
    _label = [[UILabel alloc]initWithFrame:CGRectMake(20, 27, 110, 30)];
    _label.text = @"Click start";
    _label.font = [UIFont fontWithName:@"New York" size:22];
    _label.textColor = [UIColor whiteColor];
    [self.view addSubview:_label];
    
    [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<int>(kNumHands))
                               named:kNumHandsInputSidePacket];
    [self.mediapipeGraph addFrameOutputStream:kLandmarksOutputStream
                           outputPacketType:MPPPacketTypeRaw];
}

#pragma mark - MPPGraphDelegate methods

// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {

  if (streamName == kLandmarksOutputStream) {
    if (packet.IsEmpty()) {
      NSLog(@"[TS:%lld] No hand landmarks", packet.Timestamp().Value());
      return;
    }
        
    const auto& multiHandLandmarks = packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();

    for (int handIndex = 0; handIndex < multiHandLandmarks.size(); ++handIndex) {
      const auto& landmarks = multiHandLandmarks[handIndex];
//      NSLog(@"\tNumber of landmarks for hand[%d]: %d", handIndex, landmarks.landmark_size());
        if (self.isRecording) {
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
    //            NSLog(@"\t\tLandmark[%d]: (%f, %f, %f)", i, landmarks.landmark(i).x(),
    //                  landmarks.landmark(i).y(), landmarks.landmark(i).z());
                
                // Add 63 points to NSArray
                [_data addObjectsFromArray:@[
                      [NSNumber numberWithFloat:landmarks.landmark(i).x()],
                      [NSNumber numberWithFloat:landmarks.landmark(i).y()],
                      [NSNumber numberWithFloat:landmarks.landmark(i).z()]]];
            }
            NSLog(@"Count %ld", [_data count]);
        }
        if ([_data count] > 0 && !self.isRecording) {
            NSData* d = [self prepareData:_data];
            _predictLabel = [_model predict:d];
            dispatch_async(dispatch_get_main_queue(), ^{
                NSLog(@"inside dispatch async block main thread from main thread");
                _label.text =  [self.labels objectAtIndex:self.predictLabel];
            });
            
            [_data removeAllObjects];
        }
     }
   }
}

- (NSData*)prepareData:(NSMutableArray*) array {
    int count = [array count];
    int diff = abs(int(count - 1134)) / 63;
    BOOL front = YES;
    if (count > 1134) {
        for(int i = 0; i < diff; i++) {
            if (front) {
                [array removeObjectsInRange:NSMakeRange(0, 63)];
                front = NO;
            } else {
                [array removeObjectsInRange:NSMakeRange([array count]-64, 63)];
                front = YES;
            }
        }
    } else {
        for(int i = 0; i < diff; i++) {
            if (front) {
                for(int j = 0; j < 63; j++) {
                    [array insertObject:[NSNumber numberWithFloat:-1.0] atIndex:0];
                }
                front = NO;
            } else {
                for(int j = 0; j < 63; j++) {
                    [array insertObject:[NSNumber numberWithFloat:-1.0] atIndex:[array count]];
                }
                front = YES;
            }
        }
    }
    NSLog(@"Array len = %i", [array count]);
//    NSLog(@"%@", array);
    int num_bytes = sizeof(float) * 1134;
    float *arr = (float*)malloc(num_bytes);
    for (int i = 0; i < 1134; ++i) {
      arr[i] = [_data[i] floatValue];
    }
    NSData *d = [NSData dataWithBytesNoCopy:arr length:num_bytes freeWhenDone:YES];
    return d;
}

- (void)recordButtonClicked:(UIButton*) sender {
    if (self.isRecording) {
        [sender setTitle:@"Start" forState:UIControlStateNormal];

        NSLog(@"%i", _predictLabel);
        _isRecording = NO;
    } else {
        [sender setTitle:@"Stop" forState:UIControlStateNormal];
        _label.text = @"Recording...";
        _isRecording = YES;
    }
}

@end
