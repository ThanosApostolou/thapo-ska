<script setup lang="ts">
import { ref } from 'vue';
import CompChat from './CompChat.vue';
import { ChatPacketType, DtoChatPacket } from './dtos/dto_chat_packet';

// hooks
const chatPackets = ref<DtoChatPacket[]>([
  new DtoChatPacket({
    timestamp: Date.now(),
    value: 'Please ask me anything related to this field',
    packet_type: ChatPacketType.ANSWER
  })
])

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
      <CompChat :chat-packets="chatPackets" />
    </div>
  </div>
</template>