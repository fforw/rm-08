// noinspection ES6UnusedImports
import STYLE from "./style.css"
import perfNow from "performance-now"
import queryString from "query-string"

import vertexShaderSource from "./rm-08.vert"
import fragmentShaderSource from "./rm-08.frag"
import Color from "./Color";

//console.log(fragmentShaderSource)

const PHI = (1 + Math.sqrt(5)) / 2;
const TAU = Math.PI * 2;
const DEG2RAD_FACTOR = TAU / 360;

const config = {
    width: 0,
    height: 0
};

let canvas, gl, vao, program;


// uniform: current time
let u_time;

let u_symmetry;

let u_resolution;

let u_mouse;

let u_palette;

let u_env;

let mouseX = 0, mouseY = 0, mouseDown, startX, startY;

// Get the container element's bounding box
let canvasBounds;

let envTexture;

function resize()
{
    const width = (window.innerWidth) & ~15;
    const height = (window.innerHeight) | 0;

    config.width = width;
    config.height = height;

    canvas.width = width;
    canvas.height = height;

    mouseX = width/2;
    mouseY = height/2;

    gl.viewport(0, 0, canvas.width, canvas.height);
}

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (success) {
        return shader;
    }

    console.error(gl.getShaderInfoLog(shader));  // eslint-disable-line
    gl.deleteShader(shader);
    return undefined;
}

function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    const success = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (success) {
        return program;
    }

    console.error(gl.getProgramInfoLog(program));  // eslint-disable-line
    gl.deleteProgram(program);
    return undefined;
}


function printError(msg)
{
    document.getElementById("out").innerHTML = "<p>" + msg + "</p>";
}


function main(time)
{
    const f = mouseDown ? 1 : -1;

    // update uniforms
    gl.uniform1f(u_time, perfNow() / 1000.0);
    gl.uniform2f(u_resolution, config.width, config.height);
    gl.uniform4f(u_mouse, mouseX, config.height - mouseY, startX * f, (config.height - startY) * f);

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    // draw
    const primitiveType = gl.TRIANGLES;
    const offset = 0;
    const count = 6;
    gl.drawArrays(primitiveType, offset, count);

    requestAnimationFrame(main);
}


window.onload = () => {
    // Get A WebGL context
    canvas = document.getElementById("screen");

    gl = canvas.getContext("webgl2");
    if (!gl) {
        canvas.parentNode.removeChild(canvas);
        printError("Cannot run shader. Your browser does not support WebGL2.");
        return;
    }


    // create GLSL shaders, upload the GLSL source, compile the shaders
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

    // Link the two shaders into a program
    program = createProgram(gl, vertexShader, fragmentShader);

    // look up where the vertex data needs to go.
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position");

    // Create a buffer and put three 2d clip space points in it
    const positionBuffer = gl.createBuffer();

    // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    const positions = [
        -1, -1,
         1, -1,
        -1, 1,
        -1, 1,
         1, 1,
         1,-1
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // Create a vertex array object (attribute state)
    vao = gl.createVertexArray();

    // and make it the one we're currently working with
    gl.bindVertexArray(vao);

    // Turn on the attribute
    gl.enableVertexAttribArray(positionAttributeLocation);

    // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
    const size = 2;          // 2 components per iteration
    const type = gl.FLOAT;   // the data is 32bit floats
    const normalize = false; // don't normalize the data
    const stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
    let offset = 0;        // start at the beginning of the buffer
    gl.vertexAttribPointer(
        positionAttributeLocation, size, type, normalize, stride, offset);


    resize();

    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    //gl.enable(gl.TEXTURE_2D);
    envTexture = gl.createTexture();

    // const envImage = document.getElementById("env");
    //
    // gl.bindTexture(gl.TEXTURE_2D, envTexture);
    // gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4096, 2048, 0, gl.RGBA, gl.UNSIGNED_BYTE, envImage);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
    // gl.generateMipmap(gl.TEXTURE_2D);
    // gl.bindTexture(gl.TEXTURE_2D, null);


    u_time = gl.getUniformLocation(program, "u_time");
    u_symmetry = gl.getUniformLocation(program, "u_symmetry");
    u_resolution = gl.getUniformLocation(program, "u_resolution");
    u_mouse = gl.getUniformLocation(program, "u_mouse");
    u_palette = gl.getUniformLocation(program, "u_palette");
    u_env = gl.getUniformLocation(program, "u_env");

    // Tell it to use our program (pair of shaders)
    gl.useProgram(program);

    const params = queryString.parse(location.search)

    const sym = +(params.sym || "5");
    console.log("SYM = " + sym);

    gl.uniform1f(u_symmetry, sym)

    // Bind the attribute/buffer set we want.
    gl.bindVertexArray(vao);

    window.addEventListener("resize", resize, true);
    canvas.addEventListener("mousemove", onMouseMove, true);
    canvas.addEventListener("mousedown", onMouseDown, true);
    document.addEventListener("mouseup", onMouseUp, true);


    window.addEventListener("touchstart", onMouseDown, true)
    window.addEventListener("touchmove", onMouseMove, true)
    window.addEventListener("touchend", onMouseUp, true)

    canvasBounds = document.getElementById("screen").getBoundingClientRect();

    const paletteArray = Color.from(
        [
            "#ff356c",
            "#b9479e",
            "#735cd2",
            "#356cff",
            "#778abd",
            "#c0ab74",
            "#ffc835",
            "#ff9b46",
            "#ff6b58",
            "#ff356c"
        ],
        1
    );


    gl.uniform3fv(u_palette, paletteArray);

    // gl.activeTexture(gl.TEXTURE0);
    // gl.bindTexture(gl.TEXTURE_2D, envTexture);
    // gl.uniform1i(u_env, 0);

    requestAnimationFrame(main)
}


// Apply the mouse event listener

function onMouseMove(ev)
{
    if (mouseDown)
    {
        mouseX = (ev.clientX - canvasBounds.left) + self.pageXOffset;
        mouseY = (ev.clientY - canvasBounds.top) + self.pageYOffset;
    }
}

function onMouseDown(ev)
{
    mouseDown = true;
    startX = (ev.clientX - canvasBounds.left) + self.pageXOffset;
    startY = (ev.clientY - canvasBounds.top) + self.pageYOffset;
    mouseX = startX;
    mouseY = startY;
}

function onMouseUp(ev)
{
    mouseDown = false;
}

