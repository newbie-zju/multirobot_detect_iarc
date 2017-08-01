#include <iostream>  
#include <fstream>  
#include <strstream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <time.h>

//-----------------------继承类----------------------------
//---------------------------------------------------------
//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，  
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问  
class MySVM : public CvSVM  
{  
public:  
	//获得SVM的决策函数中的alpha数组  
	double * get_alpha_vector()  
	{  
		return this->decision_func->alpha;  
	}  

	//获得SVM的决策函数中的rho参数,即偏移量  
	float get_rho()  
	{  
		return this->decision_func->rho;  
	}  
};  

//-------生成一个打乱的数组（从0开始的连续整数）-----------------
//---------------------------------------------------------
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

//--------根据元素大小赋予类型（0-train,1-vaild,2-test）----------------------
//---------------------------------------------------------
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
