#version 300 es
precision lowp float;

uniform float u_time;
uniform float u_symmetry;

uniform vec2 u_resolution;
uniform vec4 u_mouse;
uniform vec3 u_palette[10];
uniform sampler2D u_env;

const float pi = 3.141592653589793;
const float tau = pi * 2.0;
const float invTau = 1.0 / tau;
const float hpi = pi * 0.5;
const float phi = (1.0+sqrt(5.0))/2.0;

out vec4 outColor;

float atan2(in float y, in float x)
{
    return abs(x) > abs(y) ? hpi - atan(x,y) : atan(y,x);
}



float sdCircle( vec2 p, float r )
{
    return length(p) - r;
}
float sdBox( in vec2 p, in vec2 b )
{
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

float sdSegment( in vec2 p, in vec2 a, in vec2 b )
{
    vec2 pa = p-a, ba = b-a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

vec3 getPaletteColor(float id)
{
    int last = u_palette.length() - 1;
    //return id < float(last) ? mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id)) : u_palette[last];
    return mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id));
}


void main(void)
{
    vec2 uv = (gl_FragCoord.xy-.5*u_resolution.xy)/u_resolution.y;
    vec2 m = u_mouse.xy/u_resolution.xy;

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
    float d = sqrt( tx * tx + ty * ty) * size;


    float symAngle = tau/(u_symmetry*2.0);

    vec2 p = vec2(
        cos(a * symAngle) * d,
        sin(a * symAngle) * d
    );

    float t0  = u_time *  1.03 * 0.5;
    float t1  = u_time * -1.11 * 0.5;
    float t2  = u_time * -1.05 * 0.5;

    float result = sdCircle(p - vec2(200.0 + sin(t0) * 200.0, 200.0 + cos(t0) * 200.0), 100.0);

    result = min(result, sdSegment(p, vec2(300.0 + cos(t1) * 150.0, 300.0 + sin(t1) * 150.0), vec2(300.0 - cos(t1) * 150.0, 300.0 - sin(t1) * 150.0)));
    result = min(result, sdBox(p - vec2(100.0 + cos(t2) * 300.0, 500.0 + sin(t2) * 300.0), vec2(80.0,60.0)));
    result = min(result, sdCircle(p - vec2(700.0 + sin(t0) * 400.0, 700.0 + cos(t0) * 300.0), 100.0));

    float scale = 0.03;
    result *= scale;

    float id2 =  floor(result / 4.0);

    outColor = vec4(
        (floor(mod(result, 4.0)) >= 1.0 ? getPaletteColor(id2 + 1.0) : vec3(0)) * min(1.0, 1.4 - (d*0.35/size)),
        1.0
    );

    //outColor = vec4(1,0,1,1);
}
