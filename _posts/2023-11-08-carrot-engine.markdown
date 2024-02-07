---
layout: post
title:  "Creating my own game engine: Carrot"
date:   2023-11-08 21:01:00 +0100
categories: carrot game-engine
---

[![Sponza scene with a BMW from Need for Speed Most Wanted, and the glTF sample "Damaged Helmet"](/assets/images/Peeler/Sponza-2023-11-08.png)](/assets/images/Peeler/Sponza-2023-11-08.png)
{: .centering-container}
Peeler, Scene editor of Carrot (Click image to see original)
{: .caption}

## What?
![Oldest screenshot of Carrot](/assets/images/CarrotEngineSplashScreen.png)
{: .centering-container}
Logo of the engine, made by a friend
{: .caption}

Carrot is a game engine, made mostly for fun and learning purposes. Currently only supports Windows, and only tested on a RTX 3070 (and rarely a GTX 950M).
Here's a quick list of features:
- Scene Editor (called Peeler)
- Render-graph-based rendering
- C# scripting
- Raytraced lighting
- Skeletal animation support
- Audio support
- VR support (partial and not recently tested)
- ImGui support
- Fiber support
- Complicated build system*

\* some features might actually be a problem
{: .small-text }

## How?
### Libraries
- Vulkan: low level API. I use it because I want to learn about GPUs, and access raytracing APIs.
- Mono for C# scripting support
- [ImGui](https://github.com/ocornut/imgui) for debug UI. If you are reading a blog about a game engine, you have probably already heard about it.
- [Tracy](https://github.com/wolfpld/tracy/) for profiling (it is awesome you should check it out!)

### Tools
- C++: language I use at work, and performant enough to handle whatever I want. 
    - Why not Rust? I don't really like the syntax and I like fast prototyping. Also graphics API bindings are always necessarily outdated compared to C/C++ interfaces.
- CLion: C++ IDE made by JetBrains. I spent a lot of time programming in Java during high school and college, so I was already familiar with IDEA and its ecosystem.
- [Live++](https://liveplusplus.tech/) for hot-reloading. I discovered its existence at work, and cannot live without it now.

## Why?
Why not?

Carrot allows me to explore rendering & optimisation techniques that no other project could provide.
Also it's fun, and I'm learning a lot while doing this engine.

### If you want some history
According to my Git history, I started working on this engine on 2020/11/21, almost three years ago already!

At the time I had just finished playing through Pikmin 3 on Nintendo Switch with my roommate, and Pikmin 4 was still a vague rumour. I wanted more Pikmin. 
Therefore I decided to take matters into my own hands.

![Oldest screenshot of Carrot](/assets/images/CarrotHistory/screenshot-2021.png)
{: .centering-container}
Oldest screenshot of Carrot I still have around (Click image to see original)
{: .caption}

After getting a basic skeletal animation system going (which has been rewritten twice or thrice since), 
I decided to add support for raytracing, to learn how hardware raytracing is used.

That's where things got out of hand.
{% include youtube.html id="Psfk1YCEY-E" %}
First working version of raytraced lighting
{: .caption}

Years later, I had a basic scene editor...
{% include youtube.html id="_8yLI_iaYJY" %}
Also see the image at the top of the page for a more recent image.
{: .caption}

---
Render graph support

![Render graph debug view](/assets/images/CarrotHistory/render-graphs.png)

Support for render graphs. This is only a debug view, render graphs are only modifiable inside code.
{: .caption}

---
Finally, a few months ago (3 years after starting the project!), I finally got around to starting a prototype of a Pikmin clone:
{% include mastodon.html id="mastodon.gamedev.place/@jglrxavpok/111048250831518529" %}

All gameplay logic is handled in C#, here's an example to handle doors destroyed by Pikmin:
```csharp
using System.Collections.Generic;
using System.IO;
using Carrot;

namespace PikminClone.ECS {
    // Declares a new system for the ECS
    public class DoorDestructionSystem: LogicSystem {
        
        public DoorDestructionSystem(ulong handle) : base(handle) {
            // this system will work with entities which have both TransformComponent and DoorComponent
            AddComponent<TransformComponent>();
            AddComponent<DoorComponent>();
        }
        
        public override void Tick(double deltaTime) {
            ForEachEntity<TransformComponent, DoorComponent>((entity, transform, door) => {
                // set the door's health when it is spawned
                if (door.firstFrame) {
                    door.firstFrame = false;
                    door.health = door.MaxHealth;
                }

                // update health text
                if (door.HealthIndicator != null) {
                    TextComponent healthIndicatorText = door.HealthIndicator.GetComponent<TextComponent>();
                    if (healthIndicatorText != null) {
                        healthIndicatorText.Text = $"{(int)((door.health / door.MaxHealth) * 100)}%";
                    }
                }
                
                // if bots are attacking this door, decrease health
                double healthDecrement = 1.0f;
                foreach (var bot in door.botsAttacking) {
                    door.health -= (float)(deltaTime * healthDecrement);
                }

                // kill door entity
                if (door.health < 0.0f) {
                    door.DetachBots();
                    entity.Remove();
                }
            });
        }
    }
}
```

And there is still A LOT I want to add to my small engine.