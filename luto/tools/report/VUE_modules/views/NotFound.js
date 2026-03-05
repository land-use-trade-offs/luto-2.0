window.NotFound = {
  name: 'NotFound',
  setup() {
    return {};
  },
  template: /*html*/`
    <div class="flex flex-col items-center justify-center h-screen">
      <h1 class="text-2xl font-bold mb-4 text-center">404 Not Found</h1>
      <p class="text-center">The page you are looking for does not exist.</p>
      <p class="mt-6">
        <router-link to="/">
          <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition cursor-pointer">
            Go back to Home
          </button>
        </router-link>
      </p>
    </div>
  `,
};
