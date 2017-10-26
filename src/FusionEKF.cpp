#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Constructor
FusionEKF::FusionEKF() 
{
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_      = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0,      0,
                0,    0.0009, 0,
                0,    0,      0.09;

    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;
}

// Destructor
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) 
    {
        /**
        * initializing the state ekf_.x_ with the first measurement
        * creating initial covariance matrix P
        * Note: need to convert radar from polar to cartesian coordinates
        */
        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ << 1, 0, 0,    0,
                   0, 1, 0,    0,
                   0, 0, 1000, 0,
                   0, 0, 0, 1000;
        
        previous_timestamp_ = measurement_pack.timestamp_;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
        {
            // convert radar from polar to cartesian coordinates and initialize state
            cout<<"First masurement: RADAR"<<endl;
            float rho = measurement_pack.raw_measurements_(0);
            float phi = measurement_pack.raw_measurements_(1);
            float rho_dot = measurement_pack.raw_measurements_(2);
            ekf_.x_ << rho * cos(phi),
                       rho * sin(phi),
                       rho_dot * cos(phi),
                       rho_dot * sin(phi);
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
        {
            // initialize state
            cout<<"First masurement: LIDAR"<<endl;
            ekf_.x_ << measurement_pack.raw_measurements_(0),
                       measurement_pack.raw_measurements_(1),
                       0,
                       0;
        }

        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/
    
    // calculate elapsed time 
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;
    
    // updating the state transition matrix F according to the new elapsed time
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, dt, 0,
               0, 1, 0, dt,
               0, 0, 1, 0,
               0, 0, 0, 1;
    
    // updating the process noise covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    float noise_ax = 9.0;
    float noise_ay = 9.0;
    
    // pre-compute a set of terms to avoid repeated calculation
    float dt4_4 = pow(dt,4) / 4;
    float dt3_2 = pow(dt,3) / 2;
    float dt2 = pow(dt,2);

    ekf_.Q_ << dt4_4*noise_ax, 0, dt3_2*noise_ax, 0,
               0, dt4_4*noise_ay, 0, dt3_2*noise_ay,
               dt3_2*noise_ax, 0, dt2*noise_ax, 0, 
               0, dt3_2*noise_ay, 0, dt2*noise_ay;

    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/
    
    // using the sensor type to perform the update step 
    // updating the state and covariance matrices
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
        // radar updates
        Hj_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.H_ = Hj_;
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } 
    else 
    {
        // laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout<<"x_ = "<<ekf_.x_<<endl;
    cout<<"P_ = "<<ekf_.P_<<endl;
}
