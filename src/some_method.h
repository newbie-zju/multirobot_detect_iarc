#include <iostream>  
#include <fstream>  
#include <strstream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <time.h>
#include <algorithm>

using namespace cv;
using namespace std;

//SVM: decision_func is "protected", to get alpha and rho, you have to create a class to inherit from CvSVM
class MySVM : public CvSVM  
{  
public:  
  double * get_alpha_vector()  
  {
    return this->decision_func->alpha;  
  }
  float get_rho()  
  {
    return this->decision_func->rho;  
  }
};

//HOG: set detectHOG from detectSvm
void setHOG(MySVM &detectSvm, HOGDescriptor &detectHOG)
{

  //dimension of HOG descriptor: 
  //[(window_width-block_width)/block_stride_width+1]*[(window_height-block_height)/block_stride_height+1]*bin_number*(block_width/cell_width)*(block_height/cell_height)
  int descriptorDimDetect;
  //int descriptorDimClassify;
  descriptorDimDetect = detectSvm.get_var_count();
  //descriptorDimClassify = classifySvm.get_var_count();
  int supportVectorDetectNum = detectSvm.get_support_vector_count();
  //int supportVectorCassifyNum = classifySvm.get_support_vector_count();
  cout<<"number of Detect SVM: "<<supportVectorDetectNum<<endl;  
  //cout<<"number of Classify SVM: "<<supportVectorCassifyNum<<endl;  
  Mat alphaDetectMat = Mat::zeros(1, supportVectorDetectNum, CV_32FC1);
  //Mat alphaClassifyMat = Mat::zeros(1, supportVectorCassifyNum, CV_32FC1);
  Mat supportVectorDetectMat = Mat::zeros(supportVectorDetectNum, descriptorDimDetect, CV_32FC1); 
  //Mat supportVectorClassifyMat = Mat::zeros(supportVectorCassifyNum, descriptorDimClassify, CV_32FC1);
  Mat resultDetectMat = Mat::zeros(1, descriptorDimDetect, CV_32FC1); 
  //Mat resultClassifyMat = Mat::zeros(1, descriptorDimClassify, CV_32FC1); 

  //compute w array
  for(int i=0; i<supportVectorDetectNum; i++)
  {
    const float * pSVData = detectSvm.get_support_vector(i);
    for(int j=0; j<descriptorDimDetect; j++)  
      supportVectorDetectMat.at<float>(i,j) = pSVData[j];  
  }
  //for(int i=0; i<supportVectorCassifyNum; i++)
  //{
  //  const float * pSVData = classifySvm.get_support_vector(i);
  //  for(int j=0; j<descriptorDimClassify; j++)  
  //    supportVectorClassifyMat.at<float>(i,j) = pSVData[j];  
  //}
  double * pAlphaDetectData = detectSvm.get_alpha_vector();
  //double * pAlphaClassifyData = classifySvm.get_alpha_vector();
  for(int i=0; i<supportVectorDetectNum; i++)
    alphaDetectMat.at<float>(0,i) = pAlphaDetectData[i];  
  //for(int i=0; i<supportVectorCassifyNum; i++)
  //  alphaClassifyMat.at<float>(0,i) = pAlphaClassifyData[i];  
  resultDetectMat = -1 * alphaDetectMat * supportVectorDetectMat;//resultMat = -(alphaMat * supportVectorMat)
  //resultClassifyMat = -1 * alphaClassifyMat * supportVectorClassifyMat;//resultMat = -(alphaMat * supportVectorMat)

  //get detector for setSVMDetector(const vector<float>& detector)
  vector<float> myDetector;
  for(int i=0; i<descriptorDimDetect; i++)//add resultMat
    myDetector.push_back(resultDetectMat.at<float>(0,i));  
  myDetector.push_back(detectSvm.get_rho());//add rho  
  cout<<"dimension of detect SVM Detector (w+b): "<<myDetector.size()<<endl;

  //set SVMDetector
  detectHOG.setSVMDetector(myDetector);  

  return;
}


//Train: create a disorder array (elements are integer from zero to n-1)
void random(int a[], int n)
{
  for (int nu = 0;nu<n;nu++)
  {
    a[nu] = nu;
  }
  int index, tmp, i;
  srand(time(NULL));
  for (i = 0; i <n; i++)
  {
    index = rand() % (n - i) + i;
    if (index != i)
    {
      tmp = a[i];
      a[i] = a[index];
      a[index] = tmp;
    }
  }
}

//Train: initial type array (0-train,1-vaild,2-test)
void typeHandle(int arr[],int setNo,int trainNo,int vaildNo)
{
  for (int i = 0;i<setNo;i++)
  {
    if (arr[i]<trainNo)
    {
      arr[i] = 0;
    } 
    else if(arr[i]<trainNo + vaildNo)
    {
      arr[i] = 1;
    }
    else
    {
      arr[i] = 2;
    }
  }
}

//Label: struct for robot message
struct RobotMessage
{
  Rect location_image;//robot location on image
  Point2i center;//center of robot location on image
  int label;
  float distance_min;//minimum distance for point from this frame to last frame

  RobotMessage(){}

  RobotMessage(Rect location_image0)
  {
    location_image = location_image0;
    computerCenter();
    label = 0;
    distance_min = 10000;
  }

  ~RobotMessage(){}

  void computerCenter()
  {
    center.x = location_image.x + location_image.width / 2;
    center.y = location_image.y + location_image.height / 2;
  }
};

//Label: class for labeling robot
class LabelRobot
{
private:
  vector<RobotMessage> robots;
  vector<RobotMessage> robots_last;
  int label_max;//maximum used label
  int distance_max;//max distance from last position to this position
  int number_limit;//maximum robot number

public:
  LabelRobot()
  {
    label_max = 0;
    distance_max = 100000;
    number_limit = 10;
  }

  LabelRobot(int max_distance0, int number_limit0)
  {
    label_max = 0;
    distance_max = max_distance0;
    number_limit = number_limit0;
  }

  ~LabelRobot(){}

  void input(const vector<RobotMessage> &input_irobots)
  {
    robots.clear();
    robots.insert(robots.end(), input_irobots.begin(), input_irobots.end());
  }

  void getLabel(vector<RobotMessage> &output_robots)
  {
    labelRobot();
    output_robots.clear();
    output_robots.insert(output_robots.end(), robots.begin(), robots.end());
    robots_last.clear();
    robots_last.insert(robots_last.end(), robots.begin(), robots.end());
  }

private:
  void labelRobot()
  {
    vector<RobotMessage> robots_temp;
    robots_temp.insert(robots_temp.end(), robots.begin(), robots.end());
    vector<RobotMessage> robots_last_temp;
    robots_last_temp.insert(robots_last_temp.end(), robots_last.begin(), robots_last.end());
    vector<RobotMessage> robots_labeled;

    while(1)
    {
      if (robots_temp.size() == 0 || robots_labeled.size() >= number_limit)
      {
        break;
      }

      if (robots_last_temp.size() == 0)
      {
        for (int i = 0; i<robots_temp.size();i++)
        {
          label_max++;
          robots_temp[i].label = label_max;
        }
        robots_labeled.insert(robots_labeled.end(), robots_temp.begin(), robots_temp.end());
        break;
      }

      float distance_min = 10000;
      int robot_number_min, pair_number_i, pair_number_min;
      for (int i = 0; i < robots_temp.size(); i++)
      {
        calculateMinDistance(robots_temp[i], robots_last_temp, pair_number_i);
        if (robots_temp[i].distance_min < distance_min)
        {
          robot_number_min = i;
          pair_number_min = pair_number_i;
          distance_min = robots_temp[i].distance_min;
        }
      }

      if (distance_min > distance_max)
      {
        for (int i = 0; i<robots_temp.size();i++)
        {
          label_max++;
          robots_temp[i].label = label_max;
        }
        robots_labeled.insert(robots_labeled.end(), robots_temp.begin(), robots_temp.end());
        break;
      } 
      else
      {
        robots_temp[robot_number_min].label = robots_last_temp[pair_number_min].label;
        robots_labeled.insert(robots_labeled.end(), robots_temp[robot_number_min]);
        robots_temp.erase(robots_temp.begin() + robot_number_min);
        robots_last_temp.erase(robots_last_temp.begin() + pair_number_min);
      }
    }

    robots.clear();
    robots.insert(robots.end(), robots_labeled.begin(), robots_labeled.end());
  }

  void calculateMinDistance(RobotMessage &robots_temp_i, vector<RobotMessage> &robots_last_temp, int &pair_number_i)
  {
    float dx, dy, distance;
    for (int i = 0; i<robots_last_temp.size(); i++)
    {
      dx = robots_temp_i.center.x - robots_last_temp[i].center.x;
      dy = robots_temp_i.center.y - robots_last_temp[i].center.y;
      distance = sqrt(dx*dx+dy*dy);
      if (distance <= robots_temp_i.distance_min)
      {
        robots_temp_i.distance_min = distance;
        pair_number_i = i;
      }
    }
  }
};

//get scores
vector<float> get_scores(Mat &src, vector<Rect> &boxes, MySVM &svm, int descriptor_dim, Size WinSize, HOGDescriptor &HOG_descriptor)
{
  vector<float> scores;
  for(int i=0; i<boxes.size(); i++)  
  {
    vector<float> descriptor;
    Mat descriptor_mat(1, descriptor_dim, CV_32FC1);
    Mat src_resize;
    
    resize(src(boxes[i]), src_resize, WinSize);
    HOG_descriptor.compute(src_resize, descriptor);
    for(int j=0; j<descriptor_dim; j++)  
      descriptor_mat.at<float>(0,j) = descriptor[j];
    float score = svm.predict(descriptor_mat, true);
    score = -score;
    //cout<<score<<endl;
    scores.push_back(score);
  }
  return scores;
}

//Non-maximum suppression
struct BoxWithScore
{
  Rect box;
  float score;
  
  BoxWithScore(){}
  BoxWithScore(Rect box0, float score0)
  {
    box = box0;
    score = score0;
  }
  ~BoxWithScore(){}
};

bool less_second(const BoxWithScore & m1, const BoxWithScore & m2) {
        return m1.score < m2.score;
}

vector<Rect> sort_boxes(const vector<Rect> &boxes, const vector<float> &scores)
{
  vector<BoxWithScore> boxes_with_scores;
  vector<Rect> boxes_sort;
  for(int i = 0; i < boxes.size(); i++)
  {
    boxes_with_scores.push_back(BoxWithScore(boxes[i], scores[i]));
  }
  sort(boxes_with_scores.begin(), boxes_with_scores.end(), less_second);
  
  for(int i = 0; i < boxes.size(); i++)
  {
    boxes_sort.push_back(boxes_with_scores[i].box);
    //cout<<boxes_with_scores[i].score<<endl;
  }
  return boxes_sort;
}

float bbOverlap(const Rect& box1, const Rect& box2)  
{  
  if (box1.x > box2.x+box2.width) { return 0.0; }  
  if (box1.y > box2.y+box2.height) { return 0.0; }  
  if (box1.x+box1.width < box2.x) { return 0.0; }  
  if (box1.y+box1.height < box2.y) { return 0.0; }  
  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);  
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);  
  float intersection = colInt * rowInt;  
  float area1 = box1.width*box1.height;  
  float area2 = box2.width*box2.height;  
  return intersection / (area1 + area2 - intersection);  
}

vector<Rect> non_maximum_suppression(const vector<Rect> &boxes, const vector<float> &scores, float suppression_rate)
{
  if(boxes.size() < 2)
    return boxes;
  
  vector<Rect> boxes_sort, boxes_nms;
  boxes_sort = sort_boxes(boxes, scores);
  while(boxes_sort.size() > 0)
  {
    for(int i = boxes_sort.size() - 2; i > -1; i--)
    {
      if(bbOverlap(boxes_sort[boxes_sort.size() - 1], boxes_sort[i]) > suppression_rate)
	boxes_sort.erase(boxes_sort.begin() + i);
    }
    boxes_nms.push_back(boxes_sort[boxes_sort.size() - 1]);
    boxes_sort.pop_back();
  }
  return boxes_nms;
}



