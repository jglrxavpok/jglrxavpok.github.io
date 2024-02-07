---
layout: post
title:  "Recreating Nanite: A basic material pass"
date:   2024-02-6 20:10:28 +0100
categories: carrot game-engine rendering recreating-nanite
image: /assets/images/recreating-nanite/material-pass/cover.png
---
[![Damaged Helmet glTF sample. Top left part of image is lit, transitioning to triangle debug view in bottom right](/assets/images/recreating-nanite/material-pass/cover.png)](/assets/images/recreating-nanite/material-pass/cover.png)
{: .centering-container }

At the end of the [previous article](/2024/01/19/recreating-nanite-lod-generation.html), you could see the albedo and lighting being applied to rendered clusters.

How?

This article will be smaller than the previous one, because a basic material pass is not that complicated once cluster rendering works. Also I purposefully chose to keep the material pass simple for now.

Related commits:
- [Add cluster instance buffer, to allow to know which material to use for a given pixel](https://github.com/jglrxavpok/Carrot/commit/2643842ebe938ce7a84b9ddf8eca3ca35f6dbb4e)
- [v1 of Material pass](https://github.com/jglrxavpok/Carrot/commit/3891eed655942b81677dd098c616be229761c471)
- [Cluster rendering now outputs motion vectors](https://github.com/jglrxavpok/Carrot/commit/9167d1c24b188bec198f8b174e80df895b9bc0f4)

Summary:
- [Introduction](#introduction)
- [Rewriting cluster rendering](#rewriting-cluster-rendering)
    - [Changing data structures](#changing-data-structures)
    - [Changing rendering](#changing-rendering)
- [Writing to the GBuffer](#writing-to-the-gbuffer)
- [Improvements](#improvements)

## Introduction
My cluster rendering writes to a [visibility buffer](/2023/11/26/recreating-nanite-visibility-buffer.html), which is a 64bit UInt texture.
- The higher 32 bits are the depth of the rendered triangle.
- The lower 7 bits are the triangle index.
- The remaining bits are the instance index which is always 0 at this point of the article.

My engine, Carrot, uses a GBuffer to render opaque geometry, with the albedo, normals and emissive targets being the most important.
The goal of the material pass is to write to the GBuffer the required information to be able to view the rendered clusters.

To do so, for each pixel of the visibility buffer:
1. Find which instance and which triangle this pixel belongs to
2. Compute the corresponding UV of that pixel
3. Based on the UV, find out the normal, tangent, albedo, emissive color, metallicness and roughness.
4. Compute the motion vector of this pixel (for reprojection/temporal algorithms)
5. Write this information to the GBuffer

## Rewriting cluster rendering
### Changing data structures

Because the instance index is always 0, there is no way to determine which object a triangle is from. 
Additionally, triangle indices are not unique between two different objects, so there is no way to find the UV, normal, color of any texel at this point.

Let's fix this.

First, I am going to split `Cluster` in 3: ClusterTemplate, ClusterInstance and InstanceData.
- `ClusterTemplate` is the data related to a cluster, as loaded from the model. That means vertex buffer, index buffer, LOD, bounds, etc. It will be shared between objects which use the same origin mesh.
- `ClusterInstance` is the data related to clusters of the object. That means references to its template, and its `InstanceData` struct. Multiple ClusterInstances share the same InstanceData, because they are from the same object.
- `InstanceData` is the data per instance of the object. Outside of computing the proper transform, it will be used for motion vector and maybe for GPU culling later.

Here's what they look like:
```cpp
struct Cluster {
    VertexBuffer vertices;
    IndexBuffer indices;
    uint8_t triangleCount;
    uint32_t lod;
    mat4 transform;
};

struct ClusterInstance {
    uint32_t clusterID;
    uint32_t materialIndex;
    uint32_t instanceDataIndex;
};

struct InstanceData {
    vec4 color;
    uvec4 uuid;
    mat4 transform;

    // used for motion vectors
    mat4 lastFrameTransform;
};
```

When an object is loaded, the corresponding ClusterInstance gets created from the templates:
```cpp
struct ClustersInstanceDescription {
    Viewport* pViewport = nullptr;
    std::span<std::shared_ptr<ClustersTemplate>> templates;
    std::span<std::shared_ptr<MaterialHandle>> pMaterials; // one per template
};

std::shared_ptr<ClusterModel> ClusterManager::addModel(const ClustersInstanceDescription& desc) {
    verify(desc.templates.size() == desc.pMaterials.size(), "There must be as many templates as material handles!");

    std::uint32_t clusterCount = 0;

    for(const auto& pTemplate : desc.templates) {
        clusterCount += pTemplate->clusters.size();
    }

    // gpuInstances is a std::vector of ClusterInstance.
    // It is be memcpy-ed to a GPU buffer.
    Async::LockGuard l { accessLock };
    requireInstanceUpdate = true;
    const std::uint32_t firstInstanceID = gpuInstances.size();
    gpuInstances.resize(firstInstanceID + clusterCount);

    // each new instance will point to the original cluster, and its corresponding material
    std::uint32_t clusterIndex = 0;
    std::uint32_t templateIndex = 0;
    for(const auto& pTemplate : desc.templates) {
        for(std::size_t i = 0; i < pTemplate->clusters.size(); i++) {
            auto& gpuInstance = gpuInstances[firstInstanceID + clusterIndex];
            // materialIndex is used to find the material 
            //  inside a bindless array
            gpuInstance.materialIndex = desc.pMaterials[templateIndex]->getSlot();

            // index of the cluster (don't clone the cluster's data for each instance!)
            gpuInstance.clusterID = pTemplate->firstCluster + i;
            clusterIndex++;
        }
        templateIndex++;
    }

    // register the model for rendering, and the created handle helps
    //  the engine keep track of which models are still active
    auto pModel = models.create(std::ref(*this),
                            desc.templates,
                            desc.pMaterials,
                            desc.pViewport,
                            firstInstanceID, clusterCount);
    // each instance will point to the instance data of the ClusterModel we just created
    for(std::size_t i = 0; i < clusterCount; i++) {
        auto& gpuInstance = gpuInstances[firstInstanceID + i];
        gpuInstance.instanceDataIndex = pModel->getSlot();
    }
    return pModel;
}
```


Next I will have change how rendering is done. I start by binding the 3 buffers corresponding to templates, instances and instance data.
Then, I need to refer to each cluster instance instead of the cluster template:

```diff
    const Carrot::BufferView clusterRefs = clusterDataPerFrame[renderContext.swapchainIndex]->view;
    const Carrot::BufferView instanceRefs = instancesPerFrame[renderContext.swapchainIndex]->view;
    const Carrot::BufferView instanceDataRefs = instanceDataPerFrame[renderContext.swapchainIndex]->view;
    if(clusterRefs) {
        renderer.bindBuffer(*packet.pipeline, renderContext, clusterRefs, 0, 0);
        renderer.bindBuffer(*packet.pipeline, renderContext, instanceRefs, 0, 1);
        renderer.bindBuffer(*packet.pipeline, renderContext, instanceDataRefs, 0, 2);
    }

-   for(auto& [index, pInstance] : instances) {
-        if(auto instance = pInstance.lock()) {
+    for(const auto& [index, pInstance] : models) {
+        if(const auto instance = pInstance.lock()) {
            if(!instance->enabled) {
                continue;
            }
            if(instance->pViewport != renderContext.pViewport) {
                continue;
            }
            packet.clearPerDrawData();
            packet.unindexedDrawCommands.clear();
            packet.useInstance(instance->instanceData);
+           std::uint32_t instanceIndex = 0;

            for(const auto& pTemplate : instance->templates) {
                std::size_t clusterOffset = 0;
                for(const auto& cluster : pTemplate->clusters) {
                    if(cluster.lod == globalLOD) {
                        auto& drawCommand = packet.unindexedDrawCommands.emplace_back();
                        drawCommand.instanceCount = 1;
                        drawCommand.firstInstance = 0;
                        drawCommand.firstVertex = 0;
                        drawCommand.vertexCount = std::uint32_t(cluster.triangleCount)*3;
                        triangleCount += cluster.triangleCount;

                        GBufferDrawData drawData;
                        drawData.materialIndex = 0;
-                       drawData.uuid0 = clusterOffset + pTemplate->firstCluster;
+                       drawData.uuid0 = instance->firstInstance + instanceIndex;
                        packet.addPerDrawData(std::span{ &drawData, 1 });
                    }
                    instanceIndex++;
                    clusterOffset++;
                }
            }
            verify(instanceIndex == instance->instanceCount, "instanceIndex == instance->instanceCount");
```

### Changing rendering
With these changes, I am ready to change how the vertex shader processes its input and is aware of the new data structures:
```diff
    // ...
    layout(location = 2) in mat4 inInstanceTransform;

    layout(location = 0) out vec4 ndcPosition;
-   layout(location = 1) out flat int drawID;
-   layout(location = 2) out flat int debugInt;
+   layout(location = 1) out flat uint instanceID;

    layout(set = 0, binding = 0, scalar) buffer ClusterRef {
        Cluster clusters[];
    };

+   layout(set = 0, binding = 1, scalar) buffer ClusterInstanceRef {
+       ClusterInstance instances[];
+   };

+   layout(set = 0, binding = 2, scalar) buffer ModelDataRef {
+       InstanceData modelData[];
+   };

    void main() {
        uint drawID = gl_DrawID;

        DrawData instanceDrawData = perDrawData.drawData[perDrawDataOffsets.offset + drawID];
-       uint clusterID = instanceDrawData.uuid0;
-       debugInt = int(clusterID);
+       instanceID = instanceDrawData.uuid0;
+       uint clusterID = instances[instanceID].clusterID;
+       uint modelDataIndex = instances[instanceID].instanceDataIndex;
-       Vertex vertex = clusters[clusterID].vertices.v[clusters[clusterID].indices.i[gl_VertexIndex]];
+       Vertex vertex = clusters[clusterID].vertices.v[clusters[clusterID].indices.i[gl_VertexIndex]];
-       mat4 modelview = cbo.view * inInstanceTransform * clusters[clusterID].transform;
+       mat4 modelview = cbo.view * modelData[modelDataIndex].transform * clusters[clusterID].transform;

        // ...
    }
```
Visibility vertex shader
{: .caption :}

First, based on the input per-draw index, I have the cluster instance. Thanks to `instanceDataIndex`, I can get the instance data (`modelDataIndex` in the shader).
Finally, I use the instance's transform to compute the final model-view matrix.

## Writing to the GBuffer
At this point, I am finally ready to write the material pass!

For now, I won't support different shaders for materials. This means all materials will have to use a Physically Based Rendering model for their shading, and they won't have a choice.
I *did* say that the material pass would be basic for now. This also means I need a single shader to do the material pass.

Buckle up, this shader is a bit long so I will present it block by block.

Here's the basic outline:
1. Fetch visibility buffer
2. Write depth
3. Find instance
4. Find cluster
5. Find triangle
6. Compute UV based on pixel position and projected triangle
7. Fetch textures
8. Normal mapping
9. Motion vectors
10. Write

--- 
Preamble:
```glsl
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_shader_image_int64 : require
```

--- 
Now, my includes which define the various structures used.
Below each include is the important part of the header that is used for this particular shader.

Buffers:
```glsl
// #include <includes/buffers.glsl>

struct Vertex {
    vec4 pos;
    vec3 color;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
};

layout(buffer_reference, std140) buffer VertexBuffer {
    Vertex v[];
};

layout(buffer_reference, scalar) buffer IndexBuffer {
    uint i[];
};

struct InstanceData {
    vec4 color;
    uvec4 uuid;
    mat4 transform;
    mat4 lastFrameTransform;
};
```

Camera:
```glsl
// #include <includes/camera.glsl>

#define DEFINE_CAMERA_SET(setID)                                                \
layout(set = setID, binding = 0) uniform CameraBufferObject {                   \
    mat4 view;                                                                  \
    mat4 inverseView;                                                           \
    mat4 jitteredProjection;                                                    \
    mat4 nonJitteredProjection;                                                 \
    mat4 inverseJitteredProjection;                                             \
    mat4 inverseNonJitteredProjection;                                          \
} cbo;                                                                          \
                                                                                \
layout(set = setID, binding = 1) uniform LastFrameCameraBufferObject {          \
    mat4 view;                                                                  \
    mat4 inverseView;                                                           \
    mat4 jitteredProjection;                                                    \
    mat4 nonJitteredProjection;                                                 \
    mat4 inverseJitteredProjection;                                             \
    mat4 inverseNonJitteredProjection;                                          \
} previousFrameCBO;
```

Cluster info:
```glsl
// #include <includes/clusters.glsl>

struct Cluster {
    VertexBuffer vertices;
    IndexBuffer indices;
    uint8_t triangleCount;
    uint32_t lod;
    mat4 transform;
};

struct ClusterInstance {
    uint32_t clusterID;
    uint32_t materialIndex;
    uint32_t instanceDataIndex;
};
```

Computation of barycentrics:
For a given 3D point `p` and a given triangle defined by its vertices `a`, `b` and `c`, find the barycentric coordinates of `p` inside this triangle.
```glsl
// #include <includes/math.glsl>

vec3 barycentrics(vec3 a, vec3 b, vec3 c, vec3 p) {
    vec3 offsetA = a - p;
    vec3 offsetB = b - p;
    vec3 offsetC = c - p;

    float invTriangleArea = 2.0 / length(cross(b-a, c-a));
    float u = length(cross(p-b, p-c)) / 2.0;
    float v = length(cross(p-a, p-c)) / 2.0;
    u *= invTriangleArea;
    v *= invTriangleArea;

    return vec3(u, v, 1-u-v);
}
```

GBuffer support:
```glsl
// #include <includes/gbuffer.glsl>
struct GBuffer {
    vec4 albedo;
    vec3 viewPosition;
    mat3 viewTBN;

    uint intProperty;
    uvec4 entityID;

    float metallicness;
    float roughness;

    vec3 emissiveColor;
    vec3 motionVector;
};

// read from/write to GBuffer, handles compression

// defines a macro DEFINE_GBUFFER_INPUTS which bind all render textures used by the GBuffer
// not shown for brievity
#include "includes/gbuffer_input.glsl"

// defines 2 functions to init and output the GBuffer struct to render targets
#include "includes/gbuffer_output.glsl"
```

Material data:
```glsl
// #include "includes/materials.glsl"

struct Material {
    vec4 baseColor;

    vec3 emissiveColor;
    uint emissive;

    vec2 metallicRoughnessFactor;

    uint albedo;
    uint normalMap;
    uint metallicRoughness;
};

#define MATERIAL_SYSTEM_SET(setID)                                                                                      \
    layout(set = setID, binding = 0, scalar) buffer MaterialBuffer { Material materials[]; };                           \
    layout(set = setID, binding = 1) uniform texture2D textures[];                                                      \
    layout(set = setID, binding = 2) uniform sampler linearSampler;                                                     \
    layout(set = setID, binding = 3) uniform sampler nearestSampler;                                                    \
    layout(set = setID, binding = 4, scalar) uniform GlobalTextures {                                                   \
        uint blueNoises[64];                                                                                            \
        uint dithering;                                                                                                 \
    } globalTextures;                                                                                                   \
                                                                                                                        \
    float dither(uvec2 coords) {                                                                                        \
        const uint DITHER_SIZE = 8;                                                                                     \
        const vec2 ditherUV = vec2(coords % DITHER_SIZE) / DITHER_SIZE;                                                 \
        return texture(sampler2D(textures[globalTextures.dithering], nearestSampler), ditherUV).r;                      \
    }                                                                                                                   \
```

Finally, I can declare my inputs:
```glsl
DEFINE_GBUFFER_INPUTS(0)
MATERIAL_SYSTEM_SET(1)

// Visibility buffer rendered via cluster rendering
layout(r64ui, set = 2, binding = 0) uniform u64image2D visibilityBuffer;

// cluster data, as explained during "Changing rendering"
layout(set = 2, binding = 1, scalar) buffer ClusterRef {
    Cluster clusters[];
};

layout(set = 2, binding = 2, scalar) buffer ClusterInstanceRef {
    ClusterInstance instances[];
};

layout(set = 2, binding = 3, scalar) buffer ModelDataRef {
    InstanceData modelData[];
};
DEFINE_CAMERA_SET(3)

// UV of the texel to draw
layout(location = 0) in vec2 screenUV;
```

Whew, a lot of structures and I have not even started!

---

Now we can finally shade some pixels:
```glsl
void main() {
    uvec2 visibilityBufferImageSize = imageSize(visibilityBuffer);
    ivec2 pixelCoords = ivec2(visibilityBufferImageSize * screenUV);
    uint64_t visibilityBufferSample = imageLoad(visibilityBuffer, pixelCoords).r;

    // check if there is something at this pixel
    if(visibilityBufferSample == 0) {
        discard;
    }

    // write depth of this new pixel
    double visibilityBufferDepth = uint(0xFFFFFFFFu - (visibilityBufferSample >> 32u)) / double(0xFFFFFFFFu);
    gl_FragDepth = float(visibilityBufferDepth);

    // extract indices of cluster and instance
    uint low = uint(visibilityBufferSample);

    uint triangleIndex = low & 0x7Fu;
    uint instanceIndex = (low >> 7) & 0x1FFFFFFu;
    uint clusterID = instances[instanceIndex].clusterID;
    uint materialIndex = instances[instanceIndex].materialIndex;
```

At this point, the next step is to compute the UV of the triangle at the current pixel:
```glsl
vec2 project(mat4 modelview, vec3 p) {
    vec4 hPosition = cbo.jitteredProjection * modelview * vec4(p, 1.0);

    hPosition.xyz /= hPosition.w;
    vec2 projected = (hPosition.xy + 1.0) / 2.0;
    return projected;
}

// ....

    uint triangleIndex = low & 0x7Fu;
    uint instanceIndex = (low >> 7) & 0x1FFFFFFu;
    uint clusterID = instances[instanceIndex].clusterID;
    uint materialIndex = instances[instanceIndex].materialIndex;

    // Fetch all vertices of the current triangle
#define getVertex(n) (clusters[clusterID].vertices.v[clusters[clusterID].indices.i[(n)]])
    Vertex vA = getVertex(triangleIndex * 3 + 0);
    Vertex vB = getVertex(triangleIndex * 3 + 1);
    Vertex vC = getVertex(triangleIndex * 3 + 2);

    // Project triangle to screen
    mat4 clusterTransform = clusters[clusterID].transform;
    uint modelDataIndex = instances[instanceIndex].instanceDataIndex;
    mat4 modelTransform = modelData[modelDataIndex].transform;
    mat4 modelview = cbo.view * modelTransform * clusterTransform;

    vec2 posA = project(modelview, vA.pos.xyz);
    vec2 posB = project(modelview, vB.pos.xyz);
    vec2 posC = project(modelview, vC.pos.xyz);

    // From projected triangle, find the barycentric coordinates of the current pixel
    // This assumes that the projected coordinates matches the non-projected coordinates
    // There's probably a mathematical reason why this works, but I'll just say matrices are magic
    vec3 barycentricsInsideTriangle = barycentrics(vec3(posA, 0), vec3(posB, 0), vec3(posC, 0), vec3(screenUV, 0));

    // Finally! Compute the UV of the pixel, and its object space position
    vec2 uv = barycentricsInsideTriangle.x * vA.uv + barycentricsInsideTriangle.y * vB.uv + barycentricsInsideTriangle.z * vC.uv;
    vec3 position = barycentricsInsideTriangle.x * vA.pos.xyz + barycentricsInsideTriangle.y * vB.pos.xyz + barycentricsInsideTriangle.z * vC.pos.xyz;
```

Now it is time to fetch some textures, let's start by the albedo:
```glsl
Material material = materials[materialIndex];
uint albedoTexture = nonuniformEXT(material.albedo);
uint normalMap = nonuniformEXT(material.normalMap);
uint emissiveTexture = nonuniformEXT(material.emissive);
uint metallicRoughnessTexture = nonuniformEXT(material.metallicRoughness);
vec4 texColor = texture(sampler2D(textures[albedoTexture], linearSampler), uv);
texColor *= material.baseColor;
texColor *= modelData[modelDataIndex].color;

texColor.a = 1.0;
if(texColor.a < 0.01) {
    discard;
}

GBuffer o = initGBuffer(mat4(1.0)); // init my GBuffer struct with default values

o.albedo = vec4(texColor.rgb, texColor.a /* TODO * instanceColor.a*/);
// transparency is not handled at all by this rendering pipeline so ignore instance alpha...
// ... for now ;)
```

Let's sprinkle some normal mapping:
```glsl
vec4 hPosition = modelview * vec4(position, 1.0);
o.viewPosition = hPosition.xyz / hPosition.w;

vec3 N = barycentricsInsideTriangle.x * vA.normal + barycentricsInsideTriangle.y * vB.normal + barycentricsInsideTriangle.z * vC.normal;
vec3 T = barycentricsInsideTriangle.x * vA.tangent.xyz + barycentricsInsideTriangle.y * vB.tangent.xyz + barycentricsInsideTriangle.z * vC.tangent.xyz;

N = mat3(modelview) * N;
T = mat3(modelview) * T;
float bitangentSign = barycentricsInsideTriangle.x * vA.tangent.w + barycentricsInsideTriangle.y * vB.tangent.w + barycentricsInsideTriangle.z * vC.tangent.w;

vec3 N_ = normalize(N);
vec3 T_ = normalize(T - dot(T, N_) * N_);

vec3 B_ = normalize(bitangentSign * cross(N_, T_));

vec3 mappedNormal = texture(sampler2D(textures[normalMap], linearSampler), uv).xyz;
mappedNormal = mappedNormal * 2 -1;
mappedNormal = normalize(mappedNormal.x * T_ + mappedNormal.y * B_ + mappedNormal.z * N_);

N_ = mappedNormal;
T_ = normalize(T - dot(T, N_) * N_);
B_ = normalize(bitangentSign * cross(N_, T_));

o.viewTBN = mat3(T_, B_, N_);
```

Then, it is time for a few misc properties:
```glsl
// shading flags, here: enable lighting
o.intProperty = IntPropertiesRayTracedLighting;

// EntityID, used for picking inside editor
o.entityID = modelData[modelDataIndex].uuid;

// metallic, roughness, emissive
vec2 metallicRoughness = texture(sampler2D(textures[metallicRoughnessTexture], linearSampler), uv).bg * material.metallicRoughnessFactor;
o.metallicness = metallicRoughness.x;
o.roughness = metallicRoughness.y;
o.emissiveColor = texture(sampler2D(textures[emissiveTexture], linearSampler), uv).rgb * material.emissiveColor;
```

And one last thing: motion vectors!
```glsl
mat4 previousFrameModelTransform = modelData[modelDataIndex].lastFrameTransform;
mat4 previousFrameModelview = previousFrameCBO.view * previousFrameModelTransform * clusterTransform;
vec4 previousFrameClipPos = previousFrameModelview * vec4(position, 1.0);
vec3 previousFrameViewPosition = previousFrameClipPos.xyz / previousFrameClipPos.w;
vec4 clipPos = cbo.nonJitteredProjection * vec4(o.viewPosition, 1.0);
vec4 previousClipPos = previousFrameCBO.nonJitteredProjection * vec4(previousFrameViewPosition, 1.0);
o.motionVector = previousClipPos.xyz/previousClipPos.w - clipPos.xyz/clipPos.w;

outputGBuffer(o, mat4(1.0));
```

And with this, the shader is ready for use.


That was lot of code, so here are a few images:

[![Damaged helmet glTF sample being scaled down. When scaling down, a different LOD is selected for each cluster composing the model, which reduces the triangle count.](/assets/images/recreating-nanite/material-pass/damaged-helmet-material-pass.gif)](/assets/images/recreating-nanite/material-pass/damaged-helmet-material-pass.gif)
{: .centering-container :}
Damaged helmet glTF sample being scaled down. When scaling down, a different LOD is selected for each cluster composing the model, which reduces the triangle count.
{: .caption :}

[![Standford bunny, rendered via cluster rendering, being lit by rotating directional light](/assets/images/recreating-nanite/material-pass/bunny-material-pass.gif)](/assets/images/recreating-nanite/material-pass/bunny-material-pass.gif)
{: .centering-container :}
Standford bunny, rendered via cluster rendering, being lit by rotating directional light
{: .caption :}

My denoising algorithm implementations produce some blurry results so the output is not *that* clean, but it shows that the materials are properly applied!

## Improvements
As usual, there is still a lot of things that could be added to this material pass:
- Different shaders: for now, all materials need to be PBR materials and no artistic possibilities are exposed.
    This is do-able by having multiple passes, one per material type.
- Transparent elements: like the previous articles, nothing has been done to support transparent models.
