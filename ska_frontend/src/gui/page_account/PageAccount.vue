<script setup lang="ts">
import { ServiceAuth } from '@/domain/auth/service_auth';
import { GlobalState } from '@/domain/global_state/global_state';
import { useGlobalStore } from '@/domain/global_state/global_store';

const globalState = GlobalState.instance();
const globalStore = useGlobalStore();

async function onManageAccountClick() {
  await ServiceAuth.accountManagement();
}

</script>

<template>
  <div class="ska-page-container">
    <div class="ska-page-column-flex">
      <div class="card bg-base-100 shadow-xl">
        <!-- <figure>
          <img src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.jpg" alt="Shoes" />
        </figure> -->
        <div class="card-body">
          <h2 class="card-title">{{ globalStore.globalStore.userDetails?.username }}</h2>

          <div class="form-control">
            <label class="label cursor-pointer">
              <span class="label-text">Email</span>
              <input type="text" readonly :value="globalStore.globalStore.userDetails?.email" class="input" />
            </label>
          </div>

          <div class="form-control">
            <label class="label cursor-pointer">
              <span class="label-text">Name</span>
              <input type="text" readonly :value="globalStore.globalStore.userDetails?.given_name" class="input" />
            </label>
          </div>

          <div class="form-control">
            <label class="label cursor-pointer">
              <span class="label-text">Surname</span>
              <input type="text" readonly :value="globalStore.globalStore.userDetails?.family_name" class="input" />
            </label>
          </div>

          <div class="form-control">
            <label class="label cursor-pointer">
              <span class="label-text">Roles</span>
              <input type="text" readonly :value="globalStore.globalStore.userDetails?.roles" class="input" />
            </label>
          </div>

          <div class="card-actions justify-end">
            <button class="btn btn-primary" @click="onManageAccountClick">Manage Account</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style></style>