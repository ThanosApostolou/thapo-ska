<script setup lang="ts">
import { ref } from 'vue';
import CompChat from './CompChat.vue';

// hooks
const chat_packets = ref<string[]>([])


</script>

<template>
  <div class="flex flex-row flex-auto min-h-0">
    <div class="ska-page-column bg-base-300 max-w-64 break-words">
      <p>test aaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbccccc</p>
      <p>test</p>
      <p>
        <button class="btn">test</button>
      </p>
      <div class="flex flex-col">
      </div>
    </div>

    <div class="ska-page-column-flex flex">
      <CompChat chat-packets=chat_packets />
    </div>
  </div>
</template>