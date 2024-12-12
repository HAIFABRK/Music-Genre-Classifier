# Step 1: Use the official Nginx image as the base
FROM nginx:alpine

# Step 2: Copy the entire frontend folder (index.html, CSS, JS, etc.) to the Nginx HTML directory
COPY . /usr/share/nginx/html/

# Step 3: Expose port 80 for the Nginx server
EXPOSE 80

# Step 4: Nginx will automatically run when the container starts, so no CMD is needed
