---
layout: post
title:  "How to create a blog post"
author: Shannon Tass
description: A sample post with instructions and tips on how to create a new blog post.   
image: "/assets/images/image5.jpg"
---

## Steps for creating a new post.  

* Create a new file in the `_posts` folder called `YYYY-MM-DD-post-name.md`, where YYYY is the year (2023), MM numeric month (01-12), and DD is the numeric day of the month (01-31).  The `post-name` is a short name for the new post with `-` between words.  **You must use this name convention for all new posts.**  

*  Make the YML heading.  All pages in the site need to start with a YML heading.  For posts you should use the following header:
```
---
layout: post
title:  "Post Name"
author: Your name
description: Short yet informative description
image: /assets/images/blog-image.jpg
---
```
* For this theme, the layout should stay as `post`.   All the other fields should be updated with the information for your particular blog post.  The blog image should be a `.jpg` or `.png` file that you should add to the folder `assets/images`.  Don't make it too large or the page will take longer to load (500-800 KB is a good size).  Leave the file path as `/assets/images/` in the header area. 

* Write the body of the blog using markdown.  There are a lot of references for markdown available.  I like the [Markdown Guide](https://www.markdownguide.org) because many of the examples show both the markdown and the html code.  There are separate pages for [basic syntax](https://www.markdownguide.org/basic-syntax/), [extended syntax](https://www.markdownguide.org/extended-syntax/), and a [cheatsheet](https://www.markdownguide.org/cheat-sheet/) for quick reference. 

* You can also use html code snippets along with the markdown.  Often, using html will give you a little more control and flexibility as demonstrated below.

---
---

### Links 

To create a link (internal or external), enclose the link text in brackets (e.g., [Statistics Department]) and then follow it immediately with the URL in parentheses (e.g., (https://statistics.byu.edu)).

For example:
```
{% raw %}My favorite department at BYU is the [Statistics Department](https://statistics.byu.edu).{% endraw %}
```
My favorite department at BYU is the [Statistics Department](https://statistics.byu.edu)


If you want external links to open in a separate window, you will need to use html code with `target="_blank"` inside the `a` tag. 

For example:
```
My favorite department at BYU is the <a href="https:statistics.byu.edu" target="_blank">Statistics Department</a>
```
My favorite department at BYU is the <a href="https:statistics.byu.edu" target="_blank">Statistics Department</a>


----
----

## Internal Links and Files

If you want to have a link that points to another location on your site or if you want to include a file (such as an image or video) you must use the `site.url` and `site.baseurl` variables when making the link reference.  For example, this link to pointing to the [About]({{site.url}}/{{site.baseurl}}/about) page is coded as:
```
[About]({% raw %}{{site.url}}/{{site.baseurl}}/about){% endraw %}
```
Paths to files should also be referenced with the `site.url` and `site.baseurl` variables (see the section on **Adding Images**).

---
---

## Adding Images
*In the examples below, if your image ends with `.png` or `.JPEG`, use the appropriate extension instead of `.jpg`.*  

Images for the blog will generally but put into the `assets/images` folder.  (You can also create a subfolder for images, but you will need to include the subfolder name in the reference link.) 

Markdown syntax for including images is `![Fig Name](path/to/image)`.  For example:
```
{% raw %}![Figure]({{site.url}}/{{site.baseurl}}/assets/images/image_name.jpg){% endraw %}
```
![Figure]({{site.url}}/{{site.baseurl}}/assets/images/image5.jpg)

---
---

### Resizing images

The image I added in the previous section seems a bit large for this post.  Unfortunately,
there isn't a good way to resize images with markdown, so if you need to resize an image, use html instead of markdown and specify the width in the style parameter as follows:

```
{% raw %}<img src="{{site.url}}/{{site.baseurl}}/assets/images/image_name.jpg" alt="" style="width:300px;"/>{% endraw %}
```

(Example with width set to 300 pixels)
<img src="{{site.url}}/{{site.baseurl}}/assets/images/image5.jpg" alt="" style="width:300px;"/>


(Example with width set to 100 pixels)
<img src="{{site.url}}/{{site.baseurl}}/assets/images/image5.jpg" alt="" style="width:100px;"/>



---
---
---

## Troubleshooting

Here are some things to keep in mind if your blog appearance isn't going as you planned:

**Problem:  The blog post that I created isn't appearing**

Possible Solutions: 
  - Check your date. GitHub pages won't display blog posts with future dates
  - Check the yaml header.  If there are any special characters in any of the fields, you need to use quotes around the entire field entry.  The most common culprit is the description.  If you're having trouble, try putting quotes around the entire description

---

**Problem:  I know that I made changes to a blog post but the changes aren't appearing**

Possible Solution:
  - Check the header.  If there are any special characters in any of the fields, you need to use quotes around the entire field entry.  The most common culprit is the description.  If you're having trouble, try putting quotes around the entire description.

---

**Problem:  My entire blog has weird formatting**

Possible Solution:
  - Usually this is an address problem.  Double check your url and baseurl in the `_config` file
