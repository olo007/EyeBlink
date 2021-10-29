#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>

using namespace std;
using namespace cv;

//CECHY HAARA                            //rozne zestawy cech najlepsze wyniki dostalem dla 1 i 2

string face_cascade_name = "haarcascade_frontalface_alt.xml";
//string face_cascade_name = "haarcascade_frontalface_alt2.xml";
//string face_cascade_name = "haarcascade_frontalface_alt_tree.xml";        //slabo wykrywa
//string face_cascade_name = "haarcascade_frontalface_default.xml";         //bardzo duzo false positive
string eyes_cascade_name = "haarcascade_eye.xml";
//string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";         //lepiej sobie radzi z okularami ale poza tym podobnie


CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

string window_name = "Oczko ;)";
string window_name2 = "Face";

//string img_name = "zdj (34).jpg";

bool openclose(Mat img, vector<vector<Point> > contours, int index){                        //WEJSCIE obraz, kontury na twarzy, nr_oka
    int xL,xP,yL,yP,iL,iP;    //skrajnie polozone punkty oka wsp x,y oraz index w contours
    xL=contours[index][0].x;
    xP=contours[index][0].x;
    yP=contours[index][0].y;
    yL=contours[index][0].y;
    iL=0;
    iP=0;

    for (int i=0;i<contours[index].size();i++){
        if(contours[index][i].x>xP){
            xP=contours[index][i].x;
            yP=contours[index][i].y;
            iP=i;
        }
        if(contours[index][i].x<xL){
            xL=contours[index][i].x;
            yL=contours[index][i].y;
            iL=i;
        }
    }
    circle(img,Point(xP,yP),3,Scalar(255,255,0),1,8);
    circle(img,Point(xL,yL),3,Scalar(255,255,0),1,8);
    //szukamy srodka luku w tablicy
    int iS,xS,yS;
    //yS=face_upper.rows; //ustawiam yS  na dole obrazka

     if((contours[index].size()+iL+iP)/2<contours[index].size()){    //(size-P+L)/2+P=(size+P+L)/2
            iS=(contours[index].size()+iL+iP)/2;
      }else{
            iS=(iL+iP-contours[index].size())/2; //(size+P+L)/2 - size = (P+L-size)/2
      }

    if(contours[index][iS].y>contours[index][(iL+iP)/2].y){
        iS=(iL+iP)/2;
    }
    xS=contours[index][iS].x;
    yS=contours[index][iS].y;
    circle(img,Point(xS,yS),3,Scalar(0,255,255),1,8);
    double pom1 = xS-xL;
    pom1=(yS-yL)/pom1;
    double pom2 = xP-xL;
    pom2=(yP-yL)/pom2;
    if(pom1<pom2){
        return true;
    }else{
        return false;
    }
    //return false;
}

//==============WYKRYWANIE TWARZY===============
void detect_face( Mat img )
{

    vector<Rect> faces;                                                      //wektor na twarze
    vector<Rect> eyes;                                                       //i na oczy
    Mat gray;                                                                //obrazek w odcieniach szarosci
    Mat face_only;                                                           //wycinek na twarz
   /* Mat blur;
    GaussianBlur(img,blur,Size(3,3),3,0) ;
    addWeighted(img, 1.5, blur, -0.5, 0, img);*/
    cvtColor(img, gray, CV_BGR2GRAY );                                       //Konwersja obrazu do odcieni szarosci



    //===========WYKRYWANIE TWARZY==========
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );
    for (unsigned i=0;i<faces.size();i++){
        Rect rect_face( faces[i] );         //Kwadrat okreslający twarz
        rectangle(img, rect_face, Scalar( 0, 0, 255 ), 2, 2, 0  );          //rysowanie kwadratu
    }
    imshow(window_name, img);
    //========WYKRYWANIE OCZU=========
    for( unsigned i = 0; i < faces.size(); i++ )
    {

        face_only = gray(faces[i]);
        Mat face_upper=face_only(Rect(face_only.cols*0.15,face_only.rows/5,face_only.cols*0.7,face_only.rows*0.3));




        if(faces.size()>0){
            //adaptiveThreshold( face_only, face_only, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,21,15 );
            adaptiveThreshold( face_only, face_only, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,25,15 );

            // dilate(face_upper,face_upper,Mat());
            // erode(face_upper,face_upper,Mat());
            // Wyznaczenie konturow

        vector<vector<Point> > contours;
            vector<Point> contours_poly;
            Rect boundRect;
            findContours(face_upper,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            cvtColor(face_upper, face_upper, CV_GRAY2BGR );
            for( int i = 0; i< contours.size(); i++ )
                     {
                         approxPolyDP( Mat(contours[i]), contours_poly, 8, true );
                         boundRect = boundingRect( Mat(contours_poly) );
                         fillConvexPoly(face_upper, contours_poly, contours_poly.size() );
                        // rectangle( face_upper, boundRect.tl(), boundRect.br(), Scalar(0, 255, 255), 1, 8, 0 );

                         //drawContours( face_upper, contours, i, Scalar(0,255,255), 1, 8 );
                     }

            //zostaly wykryte kontury w gornej polowie twarzy
            int imax1=0,imax2=0,imax3=0,imax4=0;
            double S,L,BB,malinowska;
            Moments M;
            for( int i = 0; i<contours.size(); i++ ){
                S=abs(contourArea(Mat(contours[i])));
                L=arcLength( contours[i], true );
                malinowska = L/(2*sqrt(M_PI*S))-1;
                BB=S/(2*M_PI*(M.mu20+M.mu02)); //blair-bliss
                M=moments(contours[i],true);
                bool skip=false;
                for (int j=0; j<contours[i].size();j++){
                    if (contours[i][j].y<3  ) skip=true;    //odrzucam kontury stykajace sie z gorna krawedzia obrazu
                    if(contours[i][j].y>face_upper.rows-2 ) skip=true;
                  /*  if(BB>0.97){  //odrzucam okulary itp
                            skip=true;
                            drawContours( face_upper, contours, i, Scalar(255,255,0), 1, 8 );
                    }*/
                    if(malinowska>2.5){                     //odrzucam okulary itp
                        skip=true;
                        //drawContours( face_upper, contours, i, Scalar(255,255,0), 1, 8 );
                    }
                }

                //SZUKANIE NAJWIEKSZYCH KONTUROW
                if(!skip){
                if ( S > abs(contourArea(Mat(contours[imax1])))){
                    imax4=imax3;
                    imax3=imax2;
                    imax2=imax1;
                    imax1=i;
                }else{
                    if(S > abs(contourArea(Mat(contours[imax2])))){
                        imax4=imax3;
                        imax3=imax2;
                        imax2=i;
                    }else{
                        if(S > abs(contourArea(Mat(contours[imax3])))){
                            imax4=imax3;
                            imax3=i;
                        }else{
                            if(S > abs(contourArea(Mat(contours[imax4])))){
                                imax4=i;
                            }               
                        }
                    }
                }}   //  najwieksze kontury znalezione

               }

            //BRWI NAD OCZAMI

            double x1,y1, x2,y2, x3,y3, x4,y4;
            M=moments(contours[imax1],true);
            x1=M.m10/M.m00;
            M=moments(contours[imax2],true);
            x2=M.m10/M.m00;
            M=moments(contours[imax3],true);
            x3=M.m10/M.m00;
            M=moments(contours[imax4],true);
            x4=M.m10/M.m00;

           /* cout <<x1 <<" "<<y1 <<endl;
            cout <<x2 <<" "<<y2 <<endl;
            cout <<x3 <<" "<<y3 <<endl;
            cout <<x4 <<" "<<y4 <<endl;
            */

            if((x1>x2 && x1>x3)||(x1>x2 && x1>x4)||(x1>x3 && x1>x4)){     //1 po prawo
                if(!(x2>x3 && x2>x4)){
                    if(x4>x3){          //1 i 4 PRAWO
                        int pom=imax2;
                        imax2=imax4;
                        imax4=pom;
                    }else{              //1 i 3 PRAWO
                        int pom=imax2;
                        imax2=imax3;
                        imax3=pom;
                    }
                }
            }else{      //====1 LEWO
                if(!(x2>x3 && x2 >x4)){     //3 i 4 PRAWO
                    int  pom=imax2;
                    imax2=imax4;
                    imax4=pom;
                    pom=imax1;
                    imax1=imax3;
                    imax3=pom;

                }else{
                    if(x4>x3){          //2i4 PRAWO
                        int pom=imax1;
                        imax1=imax4;
                        imax4=pom;
                    }else{              //2i3 PRAWO
                        int pom=imax1;
                        imax1=imax3;
                        imax3=pom;
                    }
                }
            }       //TERAZ 1  i 2 SA PO PRAWO


            //ODROZNIAMY OCZY OD BRWI - PO POLOZENIU
            M=moments(contours[imax1],true);
            y1=M.m01/M.m00;
            M=moments(contours[imax2],true);
            y2=M.m01/M.m00;
            M=moments(contours[imax3],true);
            y3=M.m01/M.m00;
            M=moments(contours[imax4],true);
            y4=M.m01/M.m00;
            if(y1<y2){
                int pom=imax1;
                imax1=imax2;
                imax2=pom;
            }
            if(y3<y4){
                int pom=imax3;
                imax3=imax4;
                imax4=pom;
            }

            //imax1 ->oko prawe (po prawej)
            //imax2 -> brew prawa
            //imax3 -> oko lewe
            //imax4 -> brew lewa

            //brwi na niebiesko
            drawContours( face_upper, contours, imax4, Scalar(255,0,0), 1, 8 );
            drawContours( face_upper, contours, imax2, Scalar(255,0,0), 1, 8 );

            //ODROZNIAMY ZAMKNIETE OCZY OD OTWARTYCH - WSP. MALINOWSKIEJ

           //LUK GORNEJ RZESY

            //prawe oko

            bool isopen;
            isopen=openclose(face_upper,contours,imax1);


            if(isopen){
                drawContours( face_upper, contours, imax1, Scalar(0,255,0), 1, 8 );
            }else{
                drawContours( face_upper, contours, imax1, Scalar(0,0,255), 1, 8 );
            }

            //LEWE
            isopen=openclose(face_upper,contours,imax3);

            if(isopen){
                drawContours( face_upper, contours, imax3, Scalar(0,255,0), 1, 8 );
            }else{
                drawContours( face_upper, contours, imax3, Scalar(0,0,255), 1, 8 );
            }


       /*     //WSP MALINOWSKIEJ
            double open=1.6;
            double close=1.1;
            S=abs(contourArea(Mat(contours[imax1])));
            L=arcLength( contours[imax1], true );
            malinowska = L/(2*sqrt(M_PI*S))-1;
            if(malinowska>open){
                drawContours( face_upper, contours, imax1, Scalar(0,255,0), 1, 8 );
            }else{
                if(malinowska>close){
                    drawContours( face_upper, contours, imax1, Scalar(0,255,255), 1, 8 );
                }else{
                    drawContours( face_upper, contours, imax1, Scalar(0,0,255), 1, 8 );
                }
            }
            cout <<malinowska<<"\t";
            S=abs(contourArea(Mat(contours[imax3])));
            L=arcLength( contours[imax3], true );
            malinowska = L/(2*sqrt(M_PI*S))-1;
            if(malinowska>open){
                drawContours( face_upper, contours, imax3, Scalar(0,255,0), 1, 8 );
            }else{
                if(malinowska>close){
                    drawContours( face_upper, contours, imax3, Scalar(0,255,255), 1, 8 );
                }else{
                    drawContours( face_upper, contours, imax3, Scalar(0,0,255), 1, 8 );
                }
            }
            cout<<malinowska<<endl;*/

           // drawContours( face_upper, contours, imax1, Scalar(0,0,255), 1, 8 );

          //  drawContours( face_upper, contours, imax3, Scalar(0,0,255), 1, 8 );
            if(i>0){
                waitKey(0);
            }
            imshow(window_name2,face_upper);
           // waitKey(0);
        }
        //==========stara metoda

        /*eyes_cascade.detectMultiScale(face_only, eyes, 1.1, 2,  0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
      //  eyes_cascade.detectMultiScale(img, eyes, 1.1, 2,  0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
              for( unsigned j = 0; j < eyes.size(); j++ )
              {
                    Rect rect_eye( faces[i].x + eyes[j].x, faces[i].y + eyes[j].y,eyes[j].width, eyes[j].height );
                    //Rect rect_eye(eyes[i]);
                    rectangle(img, rect_eye, Scalar( 0, 255, 0 ), 2, 2, 0  );
              }*/       //stara metoda - cechy Haara
    }

   // imshow(window_name, img);                      //Pokazanie obrazka
    waitKey(0);
}

 //==========================ROZCIAGNIECIE HISTOGRAMU===============================

void histogram (Mat img){
    vector<Mat> channels;
    cvtColor( img, img, CV_BGR2YCrCb );                         //zmiana przestrzeni barw na YCrBr
    split(img, channels);                                       //rozdzielamy obraz na 3 kanaly
    normalize( channels[0], channels[0], 0, 255, CV_MINMAX);    //rozciagamy tylko histogram odpowiedzialnego za jasnosc
    merge(channels, img);                                       //laczymy kanaly w jeden obraz
    cvtColor(img, img, CV_YCrCb2BGR);

}

int main(int argc, char** argv)
{
   // cout << "CV_MAJOR = " << CV_MAJOR_VERSION << endl;
   // cout << "CV_MINOR = " << CV_MINOR_VERSION << endl;
  //  cout << "============= TEST PASSED =============" << endl;
    ifstream plik;

    Mat img;

    //======================LADOWANIE CECH HAARA========================================

    if( !face_cascade.load( face_cascade_name ) )                           //Ładowanie pliku (cechy twarzy) ze sprawdzeniem poprawnoci
    {
           cout << "Nie znaleziono pliku " << face_cascade_name << ".";
           return -2;
    }
    if( !eyes_cascade.load(eyes_cascade_name) )                             //Ładowanie pliku (cechy oczu) ze sprawdzeniem poprawnoci
       {
           cout << "Nie znaleziono pliku " << eyes_cascade_name << ".";
           return -2;
       }
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    namedWindow(window_name2, CV_WINDOW_NORMAL);
//===============================CZYTANIE LISTY ZDJEC======================================
    vector<string>lista;
    string line;
    plik.open("Names",std::ios::in);
    if(plik.good())
    {

        while(!plik.eof())
        {
            //getline(plik,line);
            plik>>line;
           // cout <<line << endl;           //wyświetlenie linii
            lista.push_back(line);

        }

        plik.close();
    }

//=================================OBROBKA ZDJEC========================================
   // imshow("Okno",img);
    int i=0;
    while (i<lista.size()){
        img = imread (lista[i]);
        while(img.rows>600){
          pyrDown(img,img);
        }

        //pyrDown(img,img);
        histogram(img);
        detect_face(img);

        i++;
    }

  //waitKey(0);
    return 0;
}
