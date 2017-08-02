#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>  
#include <fstream>  
#include <strstream>
#include <opencv2/core/core.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <opencv2/gpu/gpu.hpp>  
#include "some_method.h"
#include "parameter.h"

//#include <dji_sdk/LocalPosition.h> //dji_sdk
#include <multirobot_detect_iarc/RobotCamPos.h>

using namespace cv;
using namespace std;

class PositionEstimate
{
public:
  //node
  ros::NodeHandle nh;
  ros::NodeHandle nh_param;
  //ros::Subscriber sub_loc;//dji_sdk
  //robot and image parameter
  int robot_x, robot_y, robot_width, robot_height, robot_center_x, robot_center_y;
  double image_width, image_height;
  //LocalPosition
  bool listen_h_flag;
  double loc_h;
  //camera
  double camera_pitch;
  double fu,fv;
  //estimate
  float c2r_x;//camera 2 robot, camera frame
  float c2r_y;
  float gu, gv, pu, pv, lu, lv, alpha, yo, h;
  double robot_mycam_x, robot_mycam_y;
  
  PositionEstimate():
  nh_param("~")
  {
    //node
    //sub_loc = nh.subscribe("/dji_sdk/local_position", 10, &PositionEstimate::localPositionCallback,this);//dji_sdk
    //camera
    if(!nh_param.getParam("camera_pitch", camera_pitch))camera_pitch = 36.0;
    if(!nh_param.getParam("fu", fu))fu = 376.629954;
    if(!nh_param.getParam("fv", fv))fv = 494.151786;
    //LocalPosition
    if(!nh_param.getParam("listen_h_flag", listen_h_flag))listen_h_flag = true;
    if(!nh_param.getParam("loc_h", loc_h))loc_h = 1.5;
  }
  
  ~PositionEstimate(){}
  
  void getEstimate(int robot_x0, int robot_y0, int robot_width0, int robot_height0, double image_width0, double image_height0)
  {
    //set robot and image parameter
    robot_x = robot_x0;
    robot_y = robot_y0;
    robot_width = robot_width0;
    robot_height = robot_height0;
    image_width = image_width0;
    image_height = image_height0;
    robot_center_x = robot_x + robot_width / 2;
    robot_center_y = robot_y + robot_height / 2;
    
    //runEstimate
    if(camera_pitch>-90.0 && camera_pitch<30.0 && loc_h>0.5 && loc_h<4)
    {
      this->runEstimate();
    }
    else
    {
      cout<<"camera_pitch or loc_h is error"<<endl;
      cout<<"camera_pitch: "<<camera_pitch<<endl;
      cout<<"loc_h: "<<loc_h<<endl;
    }
  }
  /*
  void localPositionCallback(const dji_sdk::LocalPosition::ConstPtr& msg)//dji_sdk
  {
    if(listen_h_flag)
      loc_h = msg->z;
  }
  */
  
private:
  void runEstimate()
  {
    //c2r_x
    gv = image_height / 2.0;
    pv = robot_center_y;
    yo = camera_pitch / 360.0 * (2.0*M_PI);
    h = loc_h;
    
    alpha = atan((gv - pv) /fv);
    c2r_x = tan(alpha + yo + M_PI/2.0) * h;
    
    //c2r_y
    lu = robot_center_x;
    pu = image_width / 2.0;
    
    c2r_y = (lu-pu)*cos(alpha)*h / (fu*cos(alpha+yo+M_PI/2.0));
    /*
    cout<<"gv: "<<gv<<endl;
    cout<<"pv: "<<pv<<endl;
    cout<<"yo: "<<yo<<endl;
    cout<<"h: "<<h<<endl;
    cout<<"(gv - pv) /fv: "<<(gv - pv) /fv<<endl;
    cout<<"alpha: "<<alpha<<endl;
    cout<<"c2r_x: "<<c2r_x<<endl;
    cout<<"lu: "<<lu<<endl;
    cout<<"pu: "<<pu<<endl;
    cout<<"(lu-pu)*cos(alpha)*h: "<<(lu-pu)*cos(alpha)*h<<endl;
    cout<<"(fu*cos(alpha+yo)): "<<(fu*cos(alpha+yo))<<endl;
    cout<<"c2r_y: "<<c2r_y<<endl;
    */
    //publish
    robot_mycam_x = c2r_x;
    robot_mycam_y = c2r_y;
  }
};


class MultirobotDetect
{
public:
  //node
  ros::NodeHandle nh_;
  ros::NodeHandle nh_image_param;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  string subscribed_topic;
  ros::Publisher msg_pub;
  //PositionEstimate
  bool publish_msg_flag;
  PositionEstimate pe;
  multirobot_detect_iarc::RobotCamPos rcp;
  int rob_pub_num, obs_pub_num;
  //svm
  MySVM svm_detect;
  MySVM svm_classify;
  //dimension of HOG descriptor: 
    //[(window_width-block_width)/block_stride_width+1]*[(window_height-block_height)/block_stride_height+1]*bin_number*(block_width/cell_width)*(block_height/cell_height)
  int descriptor_dim_detect;
  int descriptor_dim_classify;
  int support_vector_num_detect;//number of vector, not the dimention of vector
  int support_vector_num_classify;
  Mat alpha_mat_detect;
  Mat support_vector_mat_detect;
  Mat result_mat_detect;
  //HOG descriptor
  //gpu::HOGDescriptor HOG_descriptor_detect;//gpu
  HOGDescriptor HOG_descriptor_detect;
  HOGDescriptor HOG_descriptor_classify;
  //video
  //string INPUT_VIDEO_WINDOW_NAME;
  string RESULT_VIDEO_WINDOW_NAME;
  bool show_video_flag;
  bool save_result_video_flag;
  double video_rate;
  double image_hight;
  double image_width;
  double video_delay;
  VideoWriter result_video;
  string result_video_file_name;
  //frame
  int frame_num;
  Mat src_3,src_4,dst_3;
  gpu::GpuMat src_GPU;//gpu
  vector<Rect> location_detect;
  vector<float> result_classify;
  bool save_set_flag;
  
  MultirobotDetect(PositionEstimate pe0):
  it_(nh_),//intial it_
  nh_image_param("~")
  {
    //node
    if(!nh_image_param.getParam("subscribed_topic", subscribed_topic))subscribed_topic = "/dji_sdk/image_raw";
    // Subscrive to input video feed from "/dji_sdk/image_raw" topic, imageCb is the callback function
    image_sub_ = it_.subscribe(subscribed_topic, 1, &MultirobotDetect::imageCb, this);
    msg_pub  = nh_.advertise<multirobot_detect_iarc::RobotCamPos>("/robot_cam_position", 10);
    //PositionEstimate
    pe = pe0;
    if(!nh_image_param.getParam("publish_msg_flag", publish_msg_flag))publish_msg_flag = false;
    rob_pub_num = sizeof(rcp.rob_cam_pos_x) / sizeof(rcp.rob_cam_pos_x[0]);
    obs_pub_num = sizeof(rcp.obs_cam_pos_x) / sizeof(rcp.obs_cam_pos_x[0]);
    //svm
    svm_detect.load(DetectSvmName);
    svm_classify.load(ClassifySvmName);
    descriptor_dim_detect = svm_detect.get_var_count();
    descriptor_dim_classify = svm_classify.get_var_count();
    support_vector_num_detect = svm_detect.get_support_vector_count();
    support_vector_num_classify = svm_classify.get_support_vector_count();
    alpha_mat_detect = Mat::zeros(1, support_vector_num_detect, CV_32FC1);
    support_vector_mat_detect = Mat::zeros(support_vector_num_detect, descriptor_dim_detect, CV_32FC1);
    result_mat_detect = Mat::zeros(1, descriptor_dim_detect, CV_32FC1);
    //HOG descriptor
    //HOG_descriptor_detect = gpu::HOGDescriptor(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect,1,-1,0,0.2,false,10);//gpu
    HOG_descriptor_detect = HOGDescriptor(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect,1,-1,0,0.2,false,10);
    HOG_descriptor_classify = HOGDescriptor(WinSizeClassify,BlockSizeClassify,BlockStrideClassify,CellSizeClassify,NbinsClassify);
    for(int i=0; i<support_vector_num_detect; i++) 
    {
      const float * support_vector_detect = svm_detect.get_support_vector(i);
      for(int j=0; j<descriptor_dim_detect; j++)  
        support_vector_mat_detect.at<float>(i,j) = support_vector_detect[j];  
    }
    double * alpha_detect = svm_detect.get_alpha_vector();
    for(int i=0; i<support_vector_num_detect; i++)
      alpha_mat_detect.at<float>(0,i) = alpha_detect[i];  
    result_mat_detect = -1 * alpha_mat_detect * support_vector_mat_detect;
    vector<float> detector_detect;
    for(int i=0; i<descriptor_dim_detect; i++)
      detector_detect.push_back(result_mat_detect.at<float>(0,i)); 
    detector_detect.push_back(svm_detect.get_rho());//add rho
    
    cout<<"dimension of svm detector for HOG detect(w+b):"<<detector_detect.size()<<endl;
    HOG_descriptor_detect.setSVMDetector(detector_detect);
    //video
    if(!nh_image_param.getParam("show_video_flag", show_video_flag))show_video_flag = true;
    if(show_video_flag)
    {
    //INPUT_VIDEO_WINDOW_NAME="input video";
    RESULT_VIDEO_WINDOW_NAME="detect result";
    //namedWindow(INPUT_VIDEO_WINDOW_NAME);
    namedWindow(RESULT_VIDEO_WINDOW_NAME);
    }
    if(!nh_image_param.getParam("save_result_video_flag", save_result_video_flag))save_result_video_flag = false;
    if(!nh_image_param.getParam("rate", video_rate))video_rate = 5.0;
    video_delay = 1000/video_rate;
    if(!nh_image_param.getParam("result_video_file_name", result_video_file_name))result_video_file_name = "/home/ubuntu/ros_my_workspace/src/multirobot_detect/result/a544.avi";
    //frame
    frame_num = 1;
    if(!nh_image_param.getParam("save_set_flag", save_set_flag))save_set_flag = false;
    
  }
  
  ~MultirobotDetect()
  {
    destroyAllWindows();
  }
  
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv_ptr->image.copyTo(src_3);
    //cout<<"rows"<<src_3.rows<<endl;
    //cout<<"cols"<<src_3.cols<<endl;
    
    if(frame_num == 1)
    {
      image_hight = src_3.rows;
      image_width = src_3.cols;
      result_video = VideoWriter(result_video_file_name, CV_FOURCC('M', 'J', 'P', 'G'), video_rate, Size(image_width, image_hight));
    }
    
    //frame
    cout<<"frame_num: "<<frame_num<<endl;
    frame_num++;
    
    src_3.copyTo(dst_3);
    //cvtColor(src_3,src_4,CV_BGR2BGRA);//gpu
    //src_GPU.upload(src_4);//gpu
    
    //reset
    resetState();
    
    //detect
    //HOG_descriptor_detect.detectMultiScale(src_GPU, location_detect, HitThreshold, WinStride, Size(), DetScale, 2);//gpu
    HOG_descriptor_detect.detectMultiScale(src_3, location_detect, HitThreshold, WinStride, Size(), DetScale, 2);
    
    //classfy
    for(int i=0; i<location_detect.size(); i++)  
    {
      cout<<"width:"<<location_detect[i].width<<"  height:"<<location_detect[i].height<<endl;
      vector<float> descriptor_classify;
      Mat descriptor_mat_classify(1, descriptor_dim_classify, CV_32FC1);
      Mat src_classify;
      
      resize(src_3(location_detect[i]),src_classify,WinSizeClassify);
      HOG_descriptor_classify.compute(src_classify,descriptor_classify);
      for(int j=0; j<descriptor_dim_classify; j++)  
	descriptor_mat_classify.at<float>(0,j) = descriptor_classify[j];
      float temp_result_classify = svm_classify.predict(descriptor_mat_classify);
      cout<<temp_result_classify;
      result_classify.push_back(temp_result_classify);
      
      //label the robot for (save video, show video, save set)
      if(save_result_video_flag | show_video_flag | save_set_flag)
      {
	if (temp_result_classify == 1)//irobot
	{
	  rectangle(dst_3, location_detect[i], CV_RGB(0,0,255), 3);
	  if (save_set_flag)
	  {
	    strstream ss;
	    string s;
	    ss<<ResultVideoFile_1<<1000*frame_num+i<<".jpg";
	    ss>>s;
	    imwrite(s,src_3(location_detect[i]));
	  }
	} 
	else if (temp_result_classify == 2)//obstacle
	{
	  rectangle(dst_3, location_detect[i], CV_RGB(0,255,0), 3);
	  if (save_set_flag)
	  {
	    strstream ss;
	    string s;
	    ss<<ResultVideoFile_2<<1000*frame_num+i<<".jpg";
	    ss>>s;
	    imwrite(s,src_3(location_detect[i]));
	  }
	}
	else if (temp_result_classify ==3)//background
	{
	  rectangle(dst_3, location_detect[i], Scalar(0,0,255), 3);
	  if (save_set_flag)
	  {
	    strstream ss;
	    string s;
	    ss<<ResultVideoFile_3<<1000*frame_num+i<<".jpg";
	    ss>>s;
	    imwrite(s,src_3(location_detect[i]));
	  }
	}
	else//other
	{
	  rectangle(dst_3, location_detect[i], Scalar(255,255,255), 3);
	}
      }
    }
    
    //publish msg
    if(publish_msg_flag)
    {
      cout <<"publish"<<endl;
      //set RobotCamPos
      for(int i=0; i<result_classify.size(); i++)
      {
	if (result_classify[i] == 1 && rcp.rob_num < rob_pub_num)//irobot
	{
	  rcp.exist_rob_flag = true;
	  pe.getEstimate(location_detect[i].x, location_detect[i].y, location_detect[i].width, location_detect[i].height, double(src_3.cols), double(src_3.rows));
	  rcp.rob_cam_pos_x[rcp.rob_num] = pe.robot_mycam_x;
	  rcp.rob_cam_pos_y[rcp.rob_num] = pe.robot_mycam_y;
	  rcp.rob_num++;
	}
	else if (result_classify[i] == 2 && rcp.obs_num < obs_pub_num)//obstacle
	{
	  rcp.exist_obs_flag = true;
	  pe.getEstimate(location_detect[i].x, location_detect[i].y, location_detect[i].width, location_detect[i].height, double(src_3.cols), double(src_3.rows));
	  rcp.obs_cam_pos_x[rcp.obs_num] = pe.robot_mycam_x;
	  rcp.obs_cam_pos_y[rcp.obs_num] = pe.robot_mycam_y;
	  rcp.obs_num ++;
	}
      }
      
      //publish
      msg_pub.publish(rcp);
    }
    
    //save and show video
    if(save_result_video_flag)
    {
      result_video<<dst_3;
    }
    if(show_video_flag)
    {
      //imshow(INPUT_VIDEO_WINDOW_NAME, src_3);
      imshow(RESULT_VIDEO_WINDOW_NAME, dst_3);
      waitKey(1);
    }
  }
  
  void resetState()
  { 
    //PositionEstimate
    rcp.exist_rob_flag = false;
    rcp.exist_obs_flag = false;
    rcp.rob_num = 0;
    rcp.obs_num = 0;
    for(int i = 0; i < rob_pub_num; i++)
    {
      rcp.rob_cam_pos_x[i] = 0;
      rcp.rob_cam_pos_y[i] = 0;
    }
    for(int i = 0; i < obs_pub_num; i++)
    {
      rcp.obs_cam_pos_x[i] = 0;
      rcp.obs_cam_pos_y[i] = 0;
    }
    
    //detect
    location_detect.clear();
    result_classify.clear();
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "multirobot_detect_iarc_node");//node name
  double loop_rate;
  PositionEstimate pe;//class initializing
  MultirobotDetect md(pe);
  ros::NodeHandle nh_loop_param("~");
  if(!nh_loop_param.getParam("rate", loop_rate))loop_rate = 5;//video
  ros::Rate loop_rate_class(loop_rate);//frequency: n Hz

  while(ros::ok())
    {
      ros::spinOnce();
      loop_rate_class.sleep();
    }
    ros::spin();
    return 0;
}

