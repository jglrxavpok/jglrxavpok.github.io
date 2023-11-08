---
layout: post
title:  "Setting up this blog was not easy"
date:   2023-11-08 19:35:00 +0100
categories: blog
---
So, I've had the idea of making a blog to talk about my current game engine for a few weeks, and late Monday evening I found the motivation to start.

Therefore I went online and searched for static site generators. I did not want to have to think about security and do not really care about comments. 
I wanted a very simple blog.

The first link I found on my search engine was Jekyll, and I saw something along the lines of "You can use Jekyll for free with GitHub pages". 
"Great! I'll click a few buttons and *TADAA* I have a blog" I naively thought.

Before complaining, here's a big advantage of my current setup: I can write my posts in Markdown, which is super easy to edit. And if someday I want to change the backend and/or create my
own blog software, the content *should* be portable.

## Setting up GitHub pages
That one is relatively straight forward, GitHub pages' documentation is pretty clear.

## Installing Jekyll
Same as above, I followed the documentation and everything went right, even if I think such "simple" software should have a very small list of dependencies;
but I am no web dev, and I work on game engines *- which are far from lightweight! -*, who am I to judge?

## Changing the theme
Here's where things went wrong.

As you can see, the current theme is [Merlot](https://github.com/pages-themes/merlot). I chose it because it is pretty, simple, and does not give a h@ck3rz vibe which I am not a fan of.

According to [Add theme to Pages site](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/adding-a-theme-to-your-github-pages-site-using-jekyll), you should be able to edit `_config.yml` and then:

>Add a new line to the file for the theme name.

>    To use a supported theme, type theme: THEME-NAME, replacing THEME-NAME with the name of the theme as shown in the README of the theme's repository. For a list of supported themes, see "Supported themes" on the GitHub Pages site. For example, to select the Minima theme, type `theme: minima`.
>    To use any other Jekyll theme hosted on GitHub, type `remote_theme: THEME-NAME`, replacing `THEME-NAME` with the name of the theme as shown in the README of the theme's repository.

So I typed `theme: merlot`. Which of course does not work even though the theme is supposedly supported out-of-the-box:
> jekyll 3.9.3 | Error:  The merlot theme could not be found.

Then, I tried `theme: pages-themes/merlot@v0.2.0`, which worked!

But at that point my posts no longer showed up...

### Did you know?

GitHub pages suggests to create a first post, then to change your theme, but most of the themes suggested don't support the 'post' layout by default?
Great stuff.

Thankfully adding new layouts is easy and I have been able to re-add the missing layouts (`home` and `post`) by copy pasting heavily from the default layout of Merlot, and the default layouts of Minima (the default theme when creating a Jekyll site).

## TODO
- I don't like the width of posts, it feels too narrow, I'll try increasing the default width some day.
- I think having a link to the home page at the top of the page is important. But I don't like the fact that it is blue on posts (because it is a link), and not on the home page, I'd like an uniform color, and not blue.
- Make categories clickable.