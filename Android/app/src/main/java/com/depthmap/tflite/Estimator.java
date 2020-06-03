package com.depthmap.tflite;

import android.graphics.Bitmap;

/**
 * Created by Bean on 21/05/2020.
 */

public interface Estimator {
    Bitmap estimateDepth(Bitmap bitmap);
    void close();
}
