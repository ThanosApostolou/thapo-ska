<script setup lang="ts">
import { ref } from 'vue';

// hooks
const props = defineProps<{
  chatPackets: any
}>();
const chat_packets = ref<string[]>([])


</script>

<template>
  <div class="flex flex-col flex-auto items-stretch">
    <div class="ska-page-column-flex bg-error">
      <!-- <For each=chat_packets key=|chat_packet| chat_packet.timestamp let:chat_packet>
        <Show when=move || chat_packet.packet_type.with(|packet_type| matches!(packet_type, ChatPacketType::ANSWER))>
          <div class="chat chat-start">
            <div class="chat-bubble">{chat_packet.value}</div>
          </div>
        </Show>
        <Show when=move || chat_packet.packet_type.with(|packet_type| matches!(packet_type, ChatPacketType::QUESTION))>
          <div class="chat chat-end">
            <div class="chat-bubble">{chat_packet.value}</div>
          </div>
        </Show>
      </For> -->
    </div>

    <div class="ska-page-column">
      <!-- <form class="form-control" on:submit=move |ev| on_submit(ev, global_state, global_store, question, chat_packets)>
        <label class="label w-full">
          // <span class="label-text mr-2">Ask</span>
          <input type="text" placeholder="Ask your question" class="input input-bordered input-primary w-full mr-1"
            prop:value=question on:input=move |ev| set_question(event_target_value(&ev)) />
          <button type="submit" class="btn btn-outline btn-primary">
            <img src="assets/icons/paper-airplane.svg" height="32" width="32" />
          </button>
        </label>
      </form> -->
    </div>
  </div>
</template>