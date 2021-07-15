package com.example.myapplication;

import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import androidx.appcompat.app.AppCompatActivity;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;
import com.google.protobuf.InvalidProtocolBufferException;

import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.lang.Math;
import java.util.Arrays;
import java.util.List;


/**
 * Main activity of MediaPipe example apps.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final String BINARY_GRAPH_NAME = "pose_tracking_gpu.binarypb";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME = "pose_landmarks";
    private static final int NUM_HANDS = 2;
    private static final CameraHelper.CameraFacing CAMERA_FACING = CameraHelper.CameraFacing.BACK;
    // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
    // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
    // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
    // corner, whereas MediaPipe in general assumes the image origin is at top-left.
    private static final boolean FLIP_FRAMES_VERTICALLY = true;

    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;
    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private FrameProcessor processor;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private ExternalTextureConverter converter;
    // ApplicationInfo for retrieving metadata defined in the manifest.
    private ApplicationInfo applicationInfo;
    // Handles camera access via the {@link CameraX} Jetpack support library.
    private CameraXPreviewHelper cameraHelper;

    private Module nModule;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(getContentViewLayoutResId());

        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();


        try {
            applicationInfo =
                    getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
        } catch (NameNotFoundException e) {
            Log.e(TAG, "Cannot find application info: " + e);
        }
        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this);
        eglManager = new EglManager(null);
        processor =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        BINARY_GRAPH_NAME,
                        INPUT_VIDEO_STREAM_NAME,
                        OUTPUT_VIDEO_STREAM_NAME);
        processor
                .getVideoSurfaceOutput()
                .setFlipY(FLIP_FRAMES_VERTICALLY);


        // To show verbose logging, run:
        // adb shell setprop log.tag.MainActivity VERBOSE
//        if (Log.isLoggable(TAG, Log.VERBOSE)) {
        processor.addPacketCallback(
                OUTPUT_LANDMARKS_STREAM_NAME,
                (packet) -> {
                    Log.v(TAG, "Received Pose landmarks packet.");
                    try {
                        byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                        NormalizedLandmarkList poseLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                        Log.v(TAG, "[TS:" + packet.getTimestamp() + "] " + getPoseLandmarksDebugString(poseLandmarks));
                        SurfaceHolder srh = previewDisplayView.getHolder();
                    } catch (InvalidProtocolBufferException exception) {
                        Log.e(TAG, "failed to get proto.", exception);
                    }

                }
        );
        PermissionHelper.checkAndRequestCameraPermissions(this);
        nModule = PyTorchAndroid.loadModuleFromAsset(getAssets(),"linear_jit.pt");
    }

    // Used to obtain the content view for this application. If you are extending this class, and
    // have a custom layout, override this method and return the custom layout.
    protected int getContentViewLayoutResId() {
        return R.layout.activity_main;
    }

    @Override
    protected void onResume() {
        super.onResume();
        converter =
                new ExternalTextureConverter(
                        eglManager.getContext(), 2);
        converter.setFlipY(FLIP_FRAMES_VERTICALLY);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        converter.close();

        // Hide preview display until we re-open the camera again.
        previewDisplayView.setVisibility(View.GONE);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        previewFrameTexture = surfaceTexture;

        // Make the display view visible to start showing the preview. This triggers the
        // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
        previewDisplayView.setVisibility(View.VISIBLE);
    }

    protected Size cameraTargetResolution() {
        return null; // No preference and let the camera (helper) decide.
    }

    public void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(
                surfaceTexture -> {
                    onCameraStarted(surfaceTexture);
                });
        CameraHelper.CameraFacing cameraFacing = CameraHelper.CameraFacing.BACK;
        cameraHelper.startCamera(
                this, cameraFacing, /*unusedSurfaceTexture=*/ null, cameraTargetResolution());
    }

    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    protected void onPreviewDisplaySurfaceChanged(
            SurfaceHolder holder, int format, int width, int height) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        boolean isCameraRotated = cameraHelper.isCameraRotated();

        //displaySize.getHeight();
        //displaySize.getWidth();


        // Connect the converter to the camera-preview frames as its input (via
        // previewFrameTexture), and configure the output width and height as the computed
        // display size.
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
    }

    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                                Log.d("Surface","Surface Created");

                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                                //  height 720,1280
                                Log.d("Surface","Surface Changed");
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                                Log.d("Surface","Surface destroy");
                            }

                        });

    }
    //[0.0 , 1.0]  normazlized coordinate -> image width, height
    private String getPoseLandmarksDebugString(NormalizedLandmarkList poseLandmarks) {
        String poseLandmarkStr = "Pose landmarks: " + poseLandmarks.getLandmarkCount() + "\n";
        ArrayList<PoseLandMark> poseMarkers= new ArrayList<PoseLandMark>();
        int landmarkIndex = 0;
        /*TODO:构造x,y,z的33点
        * 使用空间换时间复杂度的方式，避免了使用n^2的时间复杂度
        * */
        float[] tensor_arr_x_i = new float[]{};
        float[] tensor_arr_y_i = new float[]{};
        float[] tensor_arr_z_i = new float[]{};

        for (NormalizedLandmark landmark : poseLandmarks.getLandmarkList()) {
            PoseLandMark marker = new PoseLandMark(landmark.getX(),landmark.getY(),landmark.getVisibility());
            ++landmarkIndex;
            poseMarkers.add(marker);

            float tensor_arr_x = landmark.getX();
            float tensor_arr_y = landmark.getY();
            float tensor_arr_z = landmark.getVisibility();

            float[] tmp_array_x = new float[tensor_arr_x_i.length+1];
            float[] tmp_array_y = new float[tensor_arr_y_i.length+1];
            float[] tmp_array_z = new float[tensor_arr_z_i.length+1];

            for (int i = 0;i < tensor_arr_x_i.length;i++){
                tmp_array_x[i] = tensor_arr_x_i[i];
                tmp_array_y[i] = tensor_arr_y_i[i];
                tmp_array_z[i] = tensor_arr_z_i[i];
            }

            tmp_array_x[tensor_arr_x_i.length]=tensor_arr_x;
            tmp_array_y[tensor_arr_y_i.length]=tensor_arr_y;
            tmp_array_z[tensor_arr_z_i.length]=tensor_arr_z;

            tensor_arr_x_i=tmp_array_x;  //[33]
            tensor_arr_y_i=tmp_array_y; //[33]
            tensor_arr_z_i=tmp_array_z; //[33]

        }

        float[][] tensor_all = {tensor_arr_x_i,tensor_arr_y_i,tensor_arr_z_i}; //[3,33]
        float[][] tensor_arr_input = new float[14][3];

        for (int i=0;i<tensor_all.length;i++){
                tensor_arr_input[0][i] = tensor_all[i][0];
                tensor_arr_input[1][i] = tensor_all[i][11];
                tensor_arr_input[3][i] = tensor_all[i][13];
                tensor_arr_input[5][i] = tensor_all[i][15];
                tensor_arr_input[11][i] = tensor_all[i][27];
                tensor_arr_input[9][i] = tensor_all[i][25];
                tensor_arr_input[7][i] = tensor_all[i][23];
                tensor_arr_input[13][i] = (tensor_all[i][11] + tensor_all[i][12]) / 2;
                tensor_arr_input[2][i] = tensor_all[i][12];
                tensor_arr_input[4][i] = tensor_all[i][14];
                tensor_arr_input[6][i] = tensor_all[i][16];
                tensor_arr_input[12][i] = tensor_all[i][28];
                tensor_arr_input[10][i] = tensor_all[i][26];
                tensor_arr_input[8][i] = tensor_all[i][24];
        }

        float[] tensor_arr_input_c = new float[14*3];
        int Index = 0;
        float max_value = 0;
        float min_value = 10;
        for (int i=0;i<tensor_arr_input.length;i++){
            for (int j=0;j<tensor_arr_input[i].length;j++){
                    if ((j==0 || j==1)&& tensor_arr_input[i][j] > max_value){
                        max_value = tensor_arr_input[i][j];
                    }
                    if ((j==0 || j==1)&& tensor_arr_input[i][j] < min_value){
                        min_value = tensor_arr_input[i][j];
                    }
            }
        }

        for (int i=0;i<tensor_arr_input.length;i++){
            for (int j=0;j<tensor_arr_input[i].length;j++){
                    if ((j==0 || j==1 )&&max_value - min_value == 0) {
                        tensor_arr_input[i][j] = (tensor_arr_input[i][j] - min_value) / (max_value - min_value + 0.001f) * 2 + 1;
                    } else {
                        tensor_arr_input[i][j] = (tensor_arr_input[i][j] - min_value) / (max_value - min_value) * 2 + 1;
                    }
                    tensor_arr_input_c[Index++] = tensor_arr_input[i][j];
            }
        }

        float[] new_data = count_center_data(tensor_arr_input);

//      TODO:model init --> input([[1,14,3],[1,14,3]]) --> output([2])  双
        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(14*3);
        FloatBuffer inTensorBuffer_2 = Tensor.allocateFloatBuffer(14*3);
        for (float val:tensor_arr_input_c){
            inTensorBuffer.put((float) val);
        }
        for (float val:new_data){
            inTensorBuffer_2.put((float) val);
        }
        Tensor inTensor = Tensor.fromBlob(inTensorBuffer,new long[]{1,14,3});
        Tensor inTensor_2 = Tensor.fromBlob(inTensorBuffer_2,new long[]{1,14,3});
        final Tensor outTensor = nModule.forward(IValue.from(inTensor),IValue.from(inTensor_2)).toTensor();
        final float[] outputs = outTensor.getDataAsFloatArray();
        TextView tv = findViewById(R.id.textView);
        if (outputs[0]>outputs[1]){
            Log.e("result is","laying");
            tv.setText("laying");
        }else{
            Log.e("result is","others");
            tv.setText("others");
        }

        // Get Angle of Positions
        double rightAngle = getAngle(poseMarkers.get(16),poseMarkers.get(14),poseMarkers.get(12));
        double leftAngle = getAngle(poseMarkers.get(15),poseMarkers.get(13),poseMarkers.get(11));
        double rightKnee = getAngle(poseMarkers.get(24),poseMarkers.get(26),poseMarkers.get(28));
        double leftKnee = getAngle(poseMarkers.get(23),poseMarkers.get(25),poseMarkers.get(27));
        double rightShoulder = getAngle(poseMarkers.get(14),poseMarkers.get(12),poseMarkers.get(24));
        double leftShoulder = getAngle(poseMarkers.get(13),poseMarkers.get(11),poseMarkers.get(23));
        Log.v(TAG,"======Degree Of Position]======\n"+
                "rightAngle :"+rightAngle+"\n"+
                "leftAngle :"+leftAngle+"\n"+
                "rightHip :"+rightKnee+"\n"+
                "leftHip :"+leftKnee+"\n"+
                "rightShoulder :"+rightShoulder+"\n"+
                "leftShoulder :"+leftShoulder+"\n");
        return poseLandmarkStr;
//        return tensor_arr_input_c;
        /*
        右手腕14右肘12右肩-&gt;右臂角度
        左腕13左肘11左肩-&gt;抬起左胳膊角度
        右骨盆24，右膝盖28，右脚踝-&gt;右膝盖角度
        左骨盆25左膝盖27左脚踝-&gt;左膝角度
        右胳膊梦12右肩膀24右骨盆--&gt;右腋窝角度
        左肩23左骨盆--&gt;左腋窝角度
        */
    }
    static double getAngle(PoseLandMark firstPoint, PoseLandMark midPoint, PoseLandMark lastPoint) {
        double result =
                Math.toDegrees(
                        Math.atan2(lastPoint.getY() - midPoint.getY(),lastPoint.getX() - midPoint.getX())
                                - Math.atan2(firstPoint.getY() - midPoint.getY(),firstPoint.getX() - midPoint.getX()));
        result = Math.abs(result); // Angle should never be negative
        if (result > 180) {
            result = (360.0 - result); // Always get the acute representation of the angle
        }
        return result;
    }

    private float[] count_center_data(float[][] landmark_point){
        float[] new_data = new float[14*3];
        float[][] fin_data = new float[14][3];
        float center_x;
        float center_y;
        float center_x_sum = 0;
        float center_y_sum = 0;
        if (landmark_point.length>0){
            for (int i=0;i<landmark_point.length;i++){
                if(i==0||i==1||i==2||i==7||i==8||i==13){
                    center_x_sum = center_x_sum + landmark_point[i][0];
                    center_y_sum = center_x_sum + landmark_point[i][1];
                }

            }
            center_x = center_x_sum / 6;
            center_y = center_y_sum / 6;

            for (int j=0;j<landmark_point.length;j++){
                fin_data[j][0] = landmark_point[j][0] - center_x;
                fin_data[j][1] = landmark_point[j][1] - center_y;
                fin_data[j][2] = landmark_point[j][2];
            }
            int index = 0;
            for (int i=0;i<fin_data.length;i++){
                for (int k=0;k<fin_data[i].length;k++){
                    new_data[index++] = fin_data[i][k];
                }
            }
        }
        return new_data;
    }
}