Drinking driver recognizer

1. Apple's iOS project:
https://developer.apple.com/documentation/vision/recognizing_objects_in_live_capture
2. drinking driver ML model taken from here:
https://www.kaggle.com/code/longvan92/using-resnet-99-success-rate-on-validation-test

During an implementation, I didn't care about architecture, forc unwraps, coding optimization and other programming stuff. My goal was to achieve the result in a time-constained situation.

Implementation flow:
1. Check what Apple offers for this task
2. Understand that I need a ML model 
3. Use the proposed model from Daniel's email
4. Download a proposed model model and convert it to CoreML model (this stage took 4 evenings!). You can check the `oldpython.py` file.
5. Use the converted model in an Apple's example project (3 hours)

How to use:
1. Run on: Xcode 13.4, iPhone iOS 15.5
2. open `BreakfastFinder.xcodeproj`
3. change team
4. Run on iPhone
5. put your iPhone into "landscape left" position
6. open `img_29.jpg` photo on your Mac
7. place iPhone's camera on this picture. Observe "result" probability on your iPhone in live. You can also open `img_20.jpg` to observe the difference in probabilities.
