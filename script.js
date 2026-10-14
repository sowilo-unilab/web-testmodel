/*
==================================================
 YOLOv11 Object Detection using ONNX Runtime Web

 Function:
 - Load YOLOv11 ONNX model
 - Process uploaded image
 - Run inference
 - Decode prediction result
 - Apply NMS
 - Draw bounding boxes

 Model:
 YOLOv11 + ONNX Runtime Web

==================================================
*/


// ==================================================
// 1. Initialize HTML Elements
// ==================================================

// Input สำหรับเลือกรูปภาพ
const imageInput = document.getElementById("imageInput");

// Canvas สำหรับแสดงผล Detection
const canvas = document.getElementById("canvas");

// Context สำหรับวาดภาพและ Bounding Box
const ctx = canvas.getContext("2d");


// ตัวแปรเก็บ ONNX Model Session
let session = null;


// ==================================================
// 2. Load YOLOv11 ONNX Model
// ==================================================

async function loadModel(){


    /*
      โหลดไฟล์ best.onnx

      executionProviders:
      - wasm ใช้ CPU ของ Browser

      สามารถเปลี่ยนเป็น webgpu
      เพื่อใช้ GPU ได้
    */

    session =
    await ort.InferenceSession.create(

        "best.onnx",

        {
            executionProviders:[
                "wasm"
            ]
        }

    );


    console.log(
        "YOLOv11 ONNX Model Loaded"
    );

}

// เรียกโหลดโมเดลทันที
loadModel();


// ==================================================
// 3. Image Upload & Object Detection
// ==================================================


imageInput.addEventListener(
"change",
async(event)=>{

    const file =
    event.target.files[0];

    if(!file)
        return;

    const img =
    new Image();

    img.onload =
    async()=>{


        /*
          กำหนดขนาด Canvas
          ให้เท่ากับ Input ของ YOLO

          YOLOv11 ใช้ 640x640
        */

        canvas.width = 640;
        canvas.height = 640;



        // แสดงภาพต้นฉบับ
        ctx.drawImage(
            img,
            0,
            0,
            640,
            640
        );


        // -------------------------------
        // Prepare Image Tensor
        // -------------------------------

        const tensor =
        preprocessImage(img);

        // -------------------------------
        // Run YOLO Model
        // -------------------------------

        const feeds = {};

        /*
          ใช้ชื่อ Input
          จาก ONNX Model

        */

        feeds[
            session.inputNames[0]
        ] = tensor;

        const result =
        await session.run(feeds);

        /*
          รับ Output จาก YOLO

          เช่น:
          output0
        */

        const outputTensor =
        result[
            session.outputNames[0]
        ];

        // -------------------------------
        // Convert Output to Boxes
        // -------------------------------

        let boxes =
        postprocessOutput(
            outputTensor,
            0.35
        );

        // -------------------------------
        // Remove Duplicate Boxes
        // -------------------------------

        boxes =
        nonMaxSuppression(
            boxes,
            0.35
        );

        // -------------------------------
        // Draw Result
        // -------------------------------

        ctx.clearRect(
            0,
            0,
            640,
            640
        );

        ctx.drawImage(
            img,
            0,
            0,
            640,
            640
        );

        drawBoxes(boxes);

    };

    // Load image file

    img.src =
    URL.createObjectURL(file);

});



// ==================================================
// 4. Image Preprocessing
// Convert Image --> Tensor
// ==================================================


function preprocessImage(img){


    const size = 640;

    // สร้าง Temporary Canvas

    const tempCanvas =
    document.createElement("canvas");

    tempCanvas.width=size;
    tempCanvas.height=size;

    const tempCtx =
    tempCanvas.getContext("2d");

    // Resize image

    tempCtx.drawImage(
        img,
        0,
        0,
        size,
        size
    );

    // อ่าน Pixel

    const imageData =
    tempCtx.getImageData(
        0,
        0,
        size,
        size
    );

    const data =
    imageData.data;

    /*
      YOLO Input Format

      Original:

      H x W x RGB
      Convert:

      [1,3,640,640]

      Channel:
      R
      G
      B

    */

    const float32 =
    new Float32Array(
        3*size*size
    );

    for(
        let i=0;
        i<size*size;
        i++
    ){

        // Red Channel

        float32[i]
        =
        data[i*4]/255;

        // Green Channel

        float32[
            i+size*size
        ]
        =
        data[i*4+1]/255;

        // Blue Channel

        float32[
            i+2*size*size
        ]
        =
        data[i*4+2]/255;

    }


    return new ort.Tensor(

        "float32",

        float32,

        [
            1,
            3,
            size,
            size
        ]

    );


}


// ==================================================
// 5. YOLO Output Processing
// Convert Prediction --> Bounding Box
// ==================================================

function postprocessOutput(
    output,
    threshold=0.35
){

    const data =
    output.data;

    const dims =
    output.dims;


    /*
      YOLOv11 Output:

      [1,7,8400]


      7 =
      x,y,w,h
      +
      3 classes

    */

    const numAttrs =
    dims[1];

    const numBoxes =
    dims[2];

    const boxes=[];

    for(
        let i=0;
        i<numBoxes;
        i++
    ){

        // Bounding Box

        const x =
        data[i];

        const y =
        data[numBoxes+i];

        const w =
        data[numBoxes*2+i];


        const h =
        data[numBoxes*3+i];


        // Find Best Class

        let bestScore=0;

        let classId=0;


        for(
            let c=4;
            c<numAttrs;
            c++
        ){

            const score =
            data[numBoxes*c+i];

            if(score>bestScore){

                bestScore=score;

                classId=c-4;

            }

        }

        // Confidence Filter

        if(
            bestScore<threshold
        )
        continue;

        boxes.push({

            x:x-w/2,

            y:y-h/2,

            width:w,

            height:h,

            score:bestScore,

            classId:classId

        });
    }

    return boxes;

}

// ==================================================
// 6. Non-Max Suppression (NMS)
// Remove Overlapping Boxes
// ==================================================

function nonMaxSuppression(
    boxes,
    iouThreshold=0.35
){
    // Sort confidence

    boxes.sort(
        (a,b)=>
        b.score-a.score
    );

    const selected=[];

    while(boxes.length){

        // เลือกกล่องคะแนนสูงสุด

        const current =
        boxes.shift();

        selected.push(current);

        /*
          ลบกล่องที่ซ้อนกันมาก

          โดยใช้ IoU
        */

        boxes =
        boxes.filter(box=>{

            if(
                box.classId
                !==
                current.classId
            )
            {
                return true;
            }

            return (
                calculateIoU(
                    current,
                    box
                )
                <
                iouThreshold
            );

        });

    }

    return selected;
}


// ==================================================
// Calculate IoU
// ==================================================

function calculateIoU(a,b){

    const x1 =
    Math.max(
        a.x,
        b.x
    );

    const y1 =
    Math.max(
        a.y,
        b.y
    );

    const x2 =
    Math.min(
        a.x+a.width,
        b.x+b.width
    );

    const y2 =
    Math.min(
        a.y+a.height,
        b.y+b.height
    );

    const intersection =
    Math.max(0,x2-x1)
    *
    Math.max(0,y2-y1);

    const areaA =
    a.width*a.height;

    const areaB =
    b.width*b.height;

    return (
        intersection /
        (
            areaA+
            areaB-
            intersection
        )
    );

}


// ==================================================
// 7. Draw Bounding Boxes
// ==================================================

async function drawBoxes(boxes){

    // Load class names

    const response =
    await fetch(
        "labels.json"
    );


    const labels =
    await response.json();




    ctx.lineWidth=3;

    ctx.font=
    "18px Arial";




    boxes.forEach(box=>{


        // Draw rectangle

        ctx.strokeStyle="red";


        ctx.strokeRect(

            box.x,

            box.y,

            box.width,

            box.height

        );





        // Label Text

        const label =

        `${labels[box.classId]}
        ${(box.score*100).toFixed(1)}%`;





        ctx.fillStyle="red";


        ctx.fillText(

            label,

            box.x,

            box.y-5

        );



    });



}
