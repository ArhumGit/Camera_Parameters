import cv2
import numpy as np

def calibrate_stereo(w, h, objpoints, imgpoints_l, imgpoints_r):
    stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS , 1000, 1e-6)
    retval, A1, D1, A2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints_l, imgpoints_r,None,None,None,None, (w,h), flags=0, criteria=stereocalib_criteria)

    return (retval, (A1,D1,A2,D2, R, T, E, F))

def calc_rms_stereo(objectpoints, imgpoints_l, imgpoints_r, A1, D1, A2, D2, R, T):
    tot_error = 0
    total_points = 0

    for i, objpoints in enumerate(objectpoints):
        # calculate world <-> cam1 transformation
        _, rvec_l, tvec_l,_ = cv2.solvePnPRansac(objpoints, imgpoints_l[i], A1, D1)

        # compute reprojection error for cam1
        rp_l, _ = cv2.projectPoints(objpoints, rvec_l, tvec_l, A1, D1)
        tot_error += np.sum(np.square(np.float64(imgpoints_l[i] - rp_l)))
        total_points += len(objpoints)

        # calculate world <-> cam2 transformation
        rvec_r, tvec_r  = cv2.composeRT(rvec_l,tvec_l,cv2.Rodrigues(R)[0],T)[:2]

        # compute reprojection error for cam2
        rp_r,_ = cv2.projectPoints(objpoints, rvec_r, tvec_r, A2, D2)
        tot_error += np.square(imgpoints_r[i] - rp_r).sum()
        total_points += len(objpoints)

    mean_error = np.sqrt(tot_error/total_points)

    return mean_error
w=0
h=0
objectpoints=0
ccc=0
if __name__ == "__main__":    
    # omitted: reading values for w,h, objectPoints, imgpoints_l, imgpoints_r from file (format as expected by the OpenCV functions)
    # [...]

    rms, (A1,D1,A2,D2,R,T,_,_) = calibrate_stereo(w, h, ccc)

    print("RMS (stereo calib): {}".format(rms))

    rms_2 = calc_rms_stereo(objectpoints, imgpoints_l, imgpoints_r, A1, D1, A2, D2, R, T)    
    print("RMS (custom calculation):", rms_2)
