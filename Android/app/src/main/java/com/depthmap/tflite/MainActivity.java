package com.depthmap.tflite;

import android.graphics.Bitmap;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String MODEL_PATH = "nyu.tflite";
    private static final boolean QUANT = true;
    private static final int INPUT_SIZE_WIDTH = 640;
    private static final int INPUT_SIZE_HEIGHT = 480;

    private Estimator estimator;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDenseDepth, btnToggleCamera;
    private ImageView imageViewResult;
    private CameraView cameraView;
    private ProgressBar pgsBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraView = findViewById(R.id.cameraView);
        imageViewResult = findViewById(R.id.imageViewResult);
        textViewResult = findViewById(R.id.textViewResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        btnToggleCamera = findViewById(R.id.btnToggleCamera);
        btnDenseDepth = findViewById(R.id.btnDetectObject);
        pgsBar = findViewById(R.id.pBar);
        pgsBar.getIndeterminateDrawable().setColorFilter(0xFF000000, PorterDuff.Mode.MULTIPLY);

        cameraView.setCropOutput(true);
        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {
                Bitmap bitmap = cameraKitImage.getBitmap();
                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT, false);
                long startTime = SystemClock.uptimeMillis();
                final Bitmap depthMap = estimator.estimateDepth(bitmap);
                long endTime = SystemClock.uptimeMillis();
                float seconds = (endTime - startTime)/1000.0f;
                imageViewResult.setImageBitmap(depthMap);
                pgsBar.setVisibility(View.GONE);
                textViewResult.setText("inference "+seconds+" seconds");
            }

            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {

            }
        });

        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.toggleFacing();
            }
        });

        btnDenseDepth.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imageViewResult.setImageBitmap(null);
                imageViewResult.destroyDrawingCache();
                pgsBar.setVisibility(v.VISIBLE);
                textViewResult.setText("");
                cameraView.captureImage();
            }
        });

        initTensorFlowAndLoadModel();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                estimator.close();
            }
        });
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    estimator = TensorflowDepthEstimator.create(
                            getAssets(),
                            MODEL_PATH,
                            QUANT);
                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnDenseDepth.setVisibility(View.VISIBLE);
            }
        });
    }
}
