#version 300 es
precision highp float;

uniform float u_time;
uniform float u_symmetry;

uniform vec2 u_resolution;
uniform vec4 u_mouse;
uniform vec3 u_palette[8];
uniform float u_shiny[8];
uniform sampler2D u_env;
uniform sampler2D u_env_ref;

const float pi = 3.141592653589793;
const float tau = pi * 2.0;
const float invTau = 1.0 / tau;
const float hpi = pi * 0.5;
const float phi = (1.0+sqrt(5.0))/2.0;

vec3 toLinearFactor = vec3(2.2);
vec3 toRGBFactor = vec3(1./2.2);

out vec4 outColor;

struct opts {
    float dummy;
};

#define MAX_STEPS 200
#define MAX_DIST 1000.
#define SURF_DIST .001

#define ROT(a) mat2(cos(a), -sin(a), sin(a), cos(a))
#define SHEARX(a) mat2(1, 0, sin(a), 1)

////////////////////// NOISE

//	Simplex 3D Noise
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //  x0 = x0 - 0. + 0.0 * C
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    // Permutations
    i = mod(i, 289.0 );
    vec4 p = permute( permute( permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
    dot(p2,x2), dot(p3,x3) ) );
}

float rand(float n){return fract(sin(n) * 43758.5453123);}

// Camera helper

vec3 Camera(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
    r = normalize(
    cross(
    vec3(0, 1, 0),
    f
    )
    ),
    u = cross(f, r),
    c = p + f * z,
    i = c + uv.x*r + uv.y*u,
    d = normalize(i-p);
    return d;
}


// 2d rotation matrix helper
mat2 Rot(float a) {
    float x = cos(a);
    float y = sin(a);
    return mat2(x, -y, y, x);
}

// RAY MARCHING PRIMITIVES

float smin(float a, float b, float k) {
    float h = clamp(0.5+0.5*(b-a)/k, 0., 1.);
    return mix(b, a, h) - k*h*(1.0-h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    return length(p-c)-r;
}

float sdCylinder(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    //t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    float x = length(p-c)-r;
    float y = (abs(t-.5)-.5)*length(ab);
    float e = length(max(vec2(x, y), 0.));
    float i = min(max(x, y), 0.);

    return e+i;
}

float sdCappedCylinder( vec3 p, float h, float r )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdSphere(vec3 p, float s)
{
    return length(p)-s;
}

float sdTorus(vec3 p, vec2 r) {
    float x = length(p.xz)-r.x;
    return length(vec2(x, p.y))-r.y;
}

float sdRoundBox(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}


float sdBeam(vec3 p, vec3 c)
{
    return length(p.xz-c.xy)-c.z;
}

float dBox(vec3 p, vec3 s) {
    p = abs(p)-s;
    return length(max(p, 0.))+min(max(p.x, max(p.y, p.z)), 0.);
}

vec2 opUnion(vec2 curr, float d, float id)
{
    if (d < curr.x)
    {
        curr.x = d;
        curr.y = id;
    }

    return curr;
}

vec2 softMinUnion(vec2 curr, float d, float id)
{
    float nx = smin(curr.x, d, 0.5);
    if (nx < curr.x)
    {
        curr.x = nx;
        curr.y = id;
    }

    return curr;
}


float sdBoundingBox(vec3 p, vec3 b, float e)
{
    p = abs(p)-b;
    vec3 q = abs(p+e)-e;
    return min(min(
    length(max(vec3(p.x, q.y, q.z), 0.0))+min(max(p.x, max(q.y, q.z)), 0.0),
    length(max(vec3(q.x, p.y, q.z), 0.0))+min(max(q.x, max(p.y, q.z)), 0.0)),
    length(max(vec3(q.x, q.y, p.z), 0.0))+min(max(q.x, max(q.y, p.z)), 0.0));
}

float sdHexPrism( vec3 p, vec2 h )
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
    length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
    p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float shape(float v, float x)
{
    return x > 0.0 ? -abs(v) : abs(v);
}

const mat2 frontPlaneRot = ROT(0.05235987755982988);


const float sin60 = sin(tau/6.0);
vec2 N22 (vec2 p) {
    vec3 a = fract(p.xyx*vec3(123.34, 234.34, 345.65));
    a += dot(a, a+34.45);
    return fract(vec2(a.x*a.y, a.y*a.z));
}

float atan2(in float y, in float x)
{
    return abs(x) > abs(y) ? hpi - atan(x,y) : atan(y,x);
}
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float sdRhombus( in vec2 p, in vec2 b )
{
    vec2 q = abs(p);
    float h = clamp((-2.0*ndot(q,b)+ndot(b,b))/dot(b,b),-1.0,1.0);
    float d = length( q - 0.5*b*vec2(1.0-h,1.0+h) );
    return d * sign( q.x*b.y + q.y*b.x - b.x*b.y );
}

float sdBox( vec3 p, vec3 b )
{
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}


vec2 getDistance(vec3 p, opts o) {

    mat2 rot45 = Rot(tau * 0.125);
    mat2 rot37 = Rot(tau * 0.19);

    vec3 q = p - vec3( 0,2,0);
    q.xy *= rot45;
    q.yz *= rot37;

    vec2 result = vec2(p.y, 2.0);
    result = opUnion(result, sdBox(q, vec3(1.5)) - 0.04, 3.0);
    result = opUnion(result, sdTorus(p - vec3( 0,2,0), vec2(4.0, 1.0)), 4.0);
    return result;

}


vec2 rayMarch(vec3 ro, vec3 rd, opts o) {


    float dO = 0.;
    float id = 0.0;

    for (int i=0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd*dO;
        vec2 result = getDistance(p, o);
        float dS = result.x;
        dO += dS;
        id = result.y;
        if (dO > MAX_DIST || abs(dS) < SURF_DIST * 0.001*(dO*.125 + 1.))
        break;
    }

    return vec2(dO, id);
}

//vec3 getNormal(vec3 p, opts o) {
//    float d = getDistance(p, o).x;
//    vec2 e = vec2(.01, 0);
//
//    vec3 n = d - vec3(
//    getDistance(p-e.xyy, o).x,
//    getDistance(p-e.yxy, o).x,
//    getDistance(p-e.yyx, o).x
//    );
//
//    return normalize(n);
//}


vec3 getNormal(vec3 p, opts o)// for function f(p)
{
    const float h = 0.0001;// replace by an appropriate value
    const vec2 k = vec2(1, -1);
    return normalize(k.xyy*getDistance(p + k.xyy*h, o).x +
    k.yyx*getDistance(p + k.yyx*h,o).x +
    k.yxy*getDistance(p + k.yxy*h,o).x +
    k.xxx*getDistance(p + k.xxx*h,o).x);
}


vec3 getPaletteColor(float id)
{
    int last = u_palette.length() - 1;
    //return id < float(last) ? mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id)) : u_palette[last];
    return mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id));
}



vec3 applyFog(
in vec3  rgb,      // original color of the pixel
in float distance, // camera to point distance
in vec3  rayOri,   // camera position
in vec3  rayDir,
in vec3 p     // camera to point vector
)
{
    float pos = p.z;

    float c = 0.005;
    float b = 0.001;// + sin((pos + p.x * sin(pos * 0.27)) * 0.31 ) * 0.15 + sin(pos * 0.17 ) * 0.15;

    float fogAmount = c * exp(-rayOri.y*b) * (1.0-exp( -distance*rayDir.y*b ))/rayDir.y;
    vec3  fogColor  = vec3(1);
    return mix( rgb, fogColor, fogAmount );
}


float softshadow( in vec3 ro, in vec3 rd, float k , opts o)
{
    float res = 1.0;
    float ph = 1e20;
    for( float t=0.001; t< MAX_DIST; )
    {
        float h = getDistance(ro + rd*t, o).x;
        if( h < 0.0001 )
        return 0.0;

        float y = h * h / ( 2.0 * ph );
        float d = sqrt( h * h - y * y);
        res = min( res, k * d /max(0.0, t -y) );
        ph = h;
        t += h;
    }
    return res;
}

vec3 getBackground(in vec3 n)
{
    vec3 col = n;// * 0.5 + 0.5;

    col = col*col*col*col;

    return (mix(vec3(1,0,0), vec3(0,1,0), -n.z) +
    mix(vec3(0,0,1), vec3(1,1,0), n.y) +
    mix(vec3(0,0,0), vec3(0,1,1), n.x));
}



vec3 getEnv(in vec3 n, bool ref)
{
    // Computing longitude
    float x = atan( -n.z, -n.x ) * invTau + 0.5;

    // Computing latitude
    float y = -n.y * 0.5 + 0.5;

    if (ref)
    {
        return texture(u_env_ref, vec2(x,y)).rgb;
    }
    else
    {
        return texture(u_env, vec2(x,y)).rgb;
    }

}

void main(void)
{
    const int mirrorLimit = 5;

    vec2 ratio = vec2(1, 1);

    vec2 uv = vec2(0);
    if (u_symmetry == 0.0)
    {
        uv = (gl_FragCoord.xy * ratio - .5 * u_resolution.xy * ratio)/min(u_resolution.x, u_resolution.y);
    }
    else
    {
        float size = min(u_resolution.x, u_resolution.y);

        float tx = ((gl_FragCoord.x) / size) * 2.0 - 1.0;
        float ty = 1.0 - ((gl_FragCoord.y) / size) * 2.0;

        if (u_resolution.x < u_resolution.y)
        {
            ty -= (u_resolution.y - u_resolution.x) / size;
        }
        else
        {
            tx -= (u_resolution.x - u_resolution.y) / size;
        }

        float a0 = ((atan2(ty,tx) - pi/2.0) / tau) * u_symmetry * 2.0;
        float id = floor(a0);
        float c = mod(id,2.0);
        float a = c - fract(a0) * (c * 2.0 - 1.0);
        float dist = sqrt( tx * tx + ty * ty);

        float symAngle = tau/(u_symmetry*2.0);

        uv = vec2(
        cos(a * symAngle) * dist * 0.5,
        sin(a * symAngle) * dist * 0.5
        );

    }


    vec2 m = u_mouse.xy/u_resolution.xy;

    vec3 ro = vec3(
        0,
        8.0,
        -10.0
    );

    float rotTime = u_time * 0.05;
    float hopTime = u_time * 0.12;


    const float minY = -0.2;
    const float maxY = 0.26;

    const float cy = (minY + maxY) / 2.0;
    const float amp = maxY - cy;

    //mat2 yzRot = Rot(min(1.15,  max(-0.25, -m.y + 0.5) * pi));
    mat2 yzRot = Rot(min(maxY,  max(minY, -m.y + 0.5 + sin(hopTime) * amp)) * pi);
    //mat2 yzRot = Rot((-m.y + 0.5) * pi);
    //mat2 yzRot = Rot(max(-0.08, -m.y + 0.5) * pi);
    ro.yz *= yzRot;

    mat2 xzRot = Rot((-m.x + 0.5 + rotTime)* 7.0);

    ro.xz *= xzRot;

    vec3 lookAt = vec3(0, 2.5, 0);

    vec3 rd = Camera(uv, ro, lookAt, 1.3);

    // aproximated direction of the moon in our tabula-rasa.jpg env map

    float angle = tau * 0.5875;

    vec3 sunDir = normalize(vec3(cos(angle), 0.4, sin(angle)));

    vec3 sunColor = vec3(4.5);

    vec3 col = vec3(0,0,1);

    opts o = opts(0.0);

    vec2 result = rayMarch(ro, rd, o);
    float d = result.x;
    vec3 p = ro + rd * d;
    if (d < MAX_DIST) {

        //vec3 lightPos = vec3(-40,20,0);
        vec3 norm = getNormal(p, o);
        float id = result.y;

        int count = 0;
        while (id == 1.0 && count < mirrorLimit)
        {
            vec3 ref = reflect(rd, norm);
            result = rayMarch(p + ref * 0.00001, ref, o);

            float d2 = result.x;
            vec3 p2 = p + ref * d2;

            p = p2;
            rd = ref;
            d += d2;

            if (d2 < MAX_DIST) {
                norm = getNormal(p, o);
                id = result.y;
            }
            else
            {
                count = mirrorLimit;
            }
            count++;
        }

        if ( round(id) == 1.0 || count >= mirrorLimit)
        {
            col = getEnv(rd, false);
        }
        else if (round(id) < 5.0)
        {

            // specular
            vec3 viewDir = normalize(rd);
            vec3 reflectDir = reflect(sunDir, norm);

            // diffuse
            float diff = max(dot(norm, sunDir), 0.0);

            vec3 ref = reflect(rd, norm);

            vec3 tone = id == 2.0 ? pow(getEnv(rd, false), toLinearFactor) : getPaletteColor(id);

            float shadow = softshadow(p, sunDir, 0.7, o);

            vec3 refColor = pow(getEnv(rd, true), toLinearFactor) * 0.1;

            vec3 diffuse = diff * tone * sunColor * shadow + refColor;
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_shiny[int(id)]);
            vec3 specular = sunColor * spec * shadow;
            //col = applyFog((id == 2.0 ? diffuse : diffuse + specular), d, ro, rd, p);
            col = pow((id == 2.0 ? diffuse : diffuse + specular), toRGBFactor);
        }
        else
        {
            vec3 tone = getPaletteColor(id);
            col = tone * 5000.0/(d*d);
        }

    }
    else
    {
        col = getEnv(rd, false);
    }



    outColor = vec4(
        col,
        1.0
    );

    //outColor = vec4(1,0,1,1);
}
