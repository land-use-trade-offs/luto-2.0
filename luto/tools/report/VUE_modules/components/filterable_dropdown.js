window.FilterableDropdown = {
  props: {
    useSearch: {
      type: Boolean,
      default: true
    },
    items: {
      type: Array,
      default: () => []
    },
    selectedValue: {
      type: String,
      default: ''
    },
    placeholder: {
      type: String,
      default: 'Select item...'
    },
    searchPlaceholder: {
      type: String,
      default: 'Search...'
    }
  },
  setup(props, { emit }) {
    const { ref, inject, onMounted, watch } = Vue;
    const items = ref([]);
    const selectedItem = ref('');
    const isOpen = ref(false);
    const searchTerm = ref('');

    // Use injected global region if no items prop provided (backwards compatibility)
    const globalSelectedRegion = inject('globalSelectedRegion', null);
    
    onMounted(async () => {
      if (props.items && props.items.length > 0) {
        items.value = [...props.items];
        selectedItem.value = props.selectedValue;
      } else {
        // Legacy behavior for regions
        await window.loadScript("./data/geo/NRM_AUS.js", 'NRM_AUS');
        items.value = window.NRM_AUS.features.map(feature => feature.properties.NHT2NAME).sort();
        selectedItem.value = globalSelectedRegion?.value || '';
      }
    });

    // Watch for prop changes
    watch(() => props.selectedValue, (newValue) => {
      selectedItem.value = newValue;
    }, { immediate: true });

    watch(() => props.items, (newItems) => {
      if (newItems && newItems.length > 0) {
        items.value = [...newItems];
      }
    }, { immediate: true, deep: true });

    return {
      selectedItem,
      isOpen,
      searchTerm,
      items,
      globalSelectedRegion
    };
  },
  computed: {
    filteredItems() {
      if (!this.useSearch || !this.searchTerm) return this.items;
      return this.items.filter(item =>
        item.toLowerCase().includes(this.searchTerm.toLowerCase())
      );
    }
  },
  mounted() {
    document.addEventListener('mousedown', this.handleClickOutside);
  },
  beforeUnmount() {
    document.removeEventListener('mousedown', this.handleClickOutside);
  },
  methods: {
    handleClickOutside(event) {
      if (this.$refs.dropdownRef && !this.$refs.dropdownRef.contains(event.target)) {
        this.isOpen = false;
      }
    },
    handleItemSelect(item) {
      this.selectedItem = item;
      this.isOpen = false;
      this.searchTerm = '';
      
      // Emit change event for parent components
      this.$emit('change', item);
      
      // Legacy support: update global region if available
      if (this.globalSelectedRegion) {
        this.globalSelectedRegion = item;
      }
    },
    toggleDropdown() {
      this.isOpen = !this.isOpen;
    },
    clearSearch() {
      this.searchTerm = '';
    }
  },
  template: `
    <div class="relative py-2 px-2" ref="dropdownRef">
      <!-- Main dropdown trigger button - displays selected region or placeholder -->
      <button
        @click="toggleDropdown"
        class="w-full py-1 px-1 text-left bg-white border border-gray-300 rounded-lg shadow-sm hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
      >
        <div class="flex items-center justify-between">
          <span class="text-[0.8rem]" :class="selectedItem ? 'text-gray-900' : 'text-gray-500'">
            {{ selectedItem || placeholder }}
          </span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5 text-gray-400 transition-transform" :class="isOpen ? 'rotate-180' : ''">
            <path d="m6 9 6 6 6-6"/>
          </svg>
        </div>
      </button>

      <!-- Dropdown content - only visible when isOpen is true -->
      <div v-if="isOpen" class="absolute mr-2 z-10 mt-1 bg-white border border-gray-300 rounded-lg shadow-lg">
        <!-- Search input field with search icon -->
        <div v-if="useSearch" class="p-0.5 border-b border-gray-200 h-8">
          <div class="relative">
            <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400">
              <circle cx="11" cy="11" r="8"/>
              <path d="m21 21-4.3-4.3"/>
            </svg>
            <input
              class="text-[0.75rem]"
              type="text"
              :placeholder="searchPlaceholder"
              v-model="searchTerm"
              class="pl-10 pr-1 py-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>

        <!-- Scrollable list of filtered region items -->
        <div class="max-h-80 overflow-y-auto">
          <div class="py-0.5">
            <button
              v-for="(item, index) in filteredItems"
              :key="index"
              @click="handleItemSelect(item)"
              class="text-[0.8rem] w-full px-1 py-0.5 text-left hover:bg-blue-50 focus:bg-blue-50 focus:outline-none transition-colors"
            >
              <span class="block text-gray-900">{{ item }}</span>
            </button>
          </div>
        </div>

      </div>

      
    </div>
  `
};