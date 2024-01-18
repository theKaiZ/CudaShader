
__device__ void color(float r, float g, float b, unsigned char* buffer){
  if(r>255)
   buffer[0] = 255;
  else
  buffer[0] = (unsigned char)(r);
  if(r>255)
    buffer[1] = 255;
  else
    buffer[1] = (unsigned char)(g);
  if(r>255)
    buffer[2] = 255;
  else
  buffer[2] = (unsigned char)(b);
}

__device__ struct mat2{
  float N[2];
  float M[2];
  __device__ mat2(float a, float b, float c, float d)
  {
    N[0] = a;
    N[1] = b;
    M[0] = c;
    M[1] = d;
  }
};



__device__ struct vec2{
  float x;
  float y;
  __device__ vec2 (){}
  __device__ vec2 (float x, float y): x(x), y(y){}
  __device__ vec2 ( float a): x(a),y(a){}
  __device__ vec2 operator+(const vec2& v) const
  {
    return {v.x+x,v.y+y};
  } 
  __device__ vec2 operator-(const vec2& v) const
  {
    return {x-v.x,y-v.y};
  }
};


  //vec2 calculations except operators

__device__ float length(vec2 v){
  return sqrt(v.x*v.x + v.y*v.y);}

__device__ float dist(vec2 posa, vec2 posb){
  float dx,dy;
  dx = posa.x - posb.x;
  dy = posa.y -posb.y;
  return sqrt(dx*dx + dy*dy);}
  

__device__ vec2 abs(vec2 v){
  return {abs(v.x),abs(v.y)};}  


//vec3 can begin

__device__ struct vec3 {
  float x;
  float y;
  float z;
  //Constructors in different ways
  __device__ vec3 (){}
  __device__ vec3 (float x, float y, float z):x(x),y(y),z(z){}
  __device__ vec3(float a):x(a),y(a),z(a){}
  __device__ vec3(vec2 v, float z):x(v.x),y(v.y),z(z){}
  __device__ vec3 operator+(const vec3& a) const{
      return {a.x + x, a.y+y, a.z+z};}
  __device__ vec3 operator-(const vec3& a) const{
      return {x-a.x,y-a.y, z-a.z};}
  __device__ vec3 operator*(const vec3& a) const{
      return {a.x*x, a.y*y , a.z*z };
  }
  __device__ vec3 operator*(const float a) const{
      return {x*a, y*a, z*a};}
  __device__ vec3 operator/(const float a) const{
      return {x/a, y/a, z/a};}
      
};

__device__ vec3 cross(vec3 a, vec3 b){
  return {a.y*b.z - b.y * a.z, a.z*b.x - b.z*a.x, a.x*b.y -b.x*a.y};
}

__device__ void color(vec3 c, unsigned char* buffer){
  if (c.x < 0)
    buffer[0] = 0;
  else if (c.x > 255)
    buffer[0] = 255;
  else
    buffer[0] = (unsigned char)(c.x);
    
  if (c.y < 0)
    buffer[1] = 0;
  else if (c.y > 255)
    buffer[1] = 255;
  else
    buffer[1] = (unsigned char)(c.y);
    
  if (c.z < 0)
    buffer[2] = 0;
  else if (c.z > 255)
    buffer[2] = 255;
  else
    buffer[2] = (unsigned char)(c.z);    
    
}

__device__ float dot(vec3 a, vec3 b){
  return a.x*b.x  + a.y*b.y + a.z*b.z;}

// I is the Incident vector (direction) and N the Normal vector of the surface
__device__ vec3 reflect(vec3 I, vec3 N){
  return I - N *dot(N,I)*2;
}


//can i put this to perators????
__device__ vec3 abs(vec3 v){
  return {abs(v.x),abs(v.y),abs(v.z)};}


__device__ float length(vec3 v){
  return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);}

__device__ float dist(vec3 a, vec3 b){
  float dx,dy,dz;
  dx = a.x-b.x;
  dy = a.y-b.y;
  dz = a.z-b.z;
  return sqrt(dx*dx+dy*dy+dz*dz);}



__device__ vec3 normalize(vec3 v){
  float betrag = length(v);
  return {v.x/betrag,v.y/betrag,v.z/betrag};}
  
  
//not sure about the following 2 formulas, a max between a vector and a float seems weird to me  
__device__ float mb(float a, float mx){
 return a > mx ? a : mx; 
}

__device__ vec3 max(vec3 v, float d){
  return {mb(v.x,d),
          mb(v.y,d),
          mb(v.z,d)};}

//i still wonder what i need a 4th dimension for, but we will see
__device__ struct vec4{
  float x;
  float y;
  float z;
  float w;
  __device__ vec4(){}
  __device__ vec4(float x, float y, float z, float w):x(x),y(y),z(z),w(w){}
  __device__ vec4(vec3 v, float w): x(v.x), y(v.y), z(v.y), w(w){}
  __device__ vec4(float a):x(a),y(a),z(a),w(a){}
  __device__ vec4(vec2 v1, vec2 v2): x(v1.x),y(v1.y),z(v2.x),w(v2.y){}
};

//init global rot variable and the functions to set during runtime
//remember that __host__ function always has to be inside extern "C" to be accessed by C-types
//btw check if there might be a less dirty way to acess the __device__ variable from __host__ function
__device__ unsigned short int rot = 0;


__device__ vec2 mouse;
__device__ vec2 window;
__device__ vec2 windowD2;
//same here with the frame
__device__ float frame = 0;
__device__ float sinfr = 0;
__device__ float cosfr = 0;


__global__ void set_vec2_g(int varnum, float x, float y){
   if (varnum==0){
       window.x = x;
       window.y = y;
       windowD2.x = x/2;
       windowD2.y = y/2;
   }
   if(varnum==1){
        mouse.x = x*2;
        mouse.y = y*2;
   }

}

__global__ void set_int_g(int varnum, int value){
    if(varnum==0){
    frame = (float)value;
  sinfr = sin(frame*M_PI/180);
  cosfr = cos(frame*M_PI/180);
    }
    if(varnum==1){
    rot=value;
    }
}


extern "C"{
__host__ void set_int(int varnum, int value){
   set_int_g<<<1,1>>>(varnum, value);
   }


__host__ void set_vec2(int varnum, float x, float y){
    set_vec2_g<<<1,1>>>(varnum, x,y);}
}


//nice functionalities for floats to have in Shady programming

__device__ float fract(float f){
  return f - floor(f);}


__device__ float step(float a, float b){
   if (abs(a) < abs(b))
     return 1;
   else
     return 0;}

__device__ float clamp(float x, float minVal, float maxVal){
  return min(max(x,minVal),maxVal);}

__device__ vec2 min(vec2 v1, vec2 v2){
  if (v1.x<v2.x)
    return v1;
  else
    return v2;
  // vec2.y soll in dem Falle die Variable für "solid/glas/mirror" sein, die abstandtsfunktion und so müssen von float auf vec2 geändert werden
  //eine neue Funktion dann auch für raymarch innerhalb glas
  // und eine für spiegelung
  //alternativ eine raymarch funktion erst für glas, dann für spiegel, dann für solids erstellen
}

__device__ vec2 max(vec2 v1, vec2 v2){
  if (v1.x>v2.x)
    return v1;
  else
    return v2;
}

__device__ float mix(float v1,float v2, float a){
  return v1* (1-a) + v2*a;}



//define smin for vec2 returns....
__device__ float smin(float a, float b, float k){//smooth min, very nice
  float h = clamp(0.5 + 0.5 * (b-a)/k,0.1,1.0);
  return mix(b,a,h) - k*h*(1.0-h);}



